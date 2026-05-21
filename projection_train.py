from __future__ import annotations
import json
import random
import logging
import torch
import argparse
import os
import math
from typing import List, Dict
from functools import partial
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
except ImportError:  # pragma: no cover - older torch builds
    FSDP = None
    FullStateDictConfig = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None
    size_based_auto_wrap_policy = None
    transformer_auto_wrap_policy = None

from utils.triplet_projector import TripletProjector
from utils.latent_refinement import LatentClinicalThoughtRefiner
from utils.graph_memory import PatientGraphMemory
from utils.triplet_retrieval import TripletRetriever
from triplet_extraction import extract_triplets
from utils.gnn import load_gnn_model
from utils.triplets_to_graph import init_bica, make_text_mappers, triplets_to_graph
import helper
import wandb

from openai import OpenAI


client_kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
if os.environ.get("OPENAI_BASE_URL"):
    client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
client = OpenAI(**client_kwargs)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)

    return distributed, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def is_fsdp_model(module) -> bool:
    return FSDP is not None and isinstance(module, FSDP)


def unwrap_model(module):
    if isinstance(module, DDP):
        return module.module
    if is_fsdp_model(module):
        return module.module
    return module


def get_fsdp_auto_wrap_policy(model: torch.nn.Module):
    layer_names = set(getattr(model, "_no_split_modules", []) or [])
    layer_classes = {m.__class__ for m in model.modules() if m.__class__.__name__ in layer_names}
    if layer_classes:
        return partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
    return partial(size_based_auto_wrap_policy, min_num_params=100_000_000)


def get_fsdp_sharding_strategy(model: torch.nn.Module) -> ShardingStrategy:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    trainable_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
    if trainable_sizes and min(trainable_sizes) < world_size:
        return ShardingStrategy.NO_SHARD
    return ShardingStrategy.FULL_SHARD


def wrap_distributed_doctor(
    doctor_model: torch.nn.Module,
    raw_doctor_model: torch.nn.Module,
    *,
    distributed: bool,
    backend: str,
    local_rank: int,
):
    if not distributed:
        return doctor_model
    if backend == "fsdp":
        if FSDP is None:
            raise RuntimeError("PyTorch FSDP is not available in this environment.")
        fsdp_kwargs = {
            "auto_wrap_policy": get_fsdp_auto_wrap_policy(raw_doctor_model),
            "sharding_strategy": get_fsdp_sharding_strategy(raw_doctor_model),
            "use_orig_params": True,
            "limit_all_gathers": True,
        }
        if torch.cuda.is_available():
            fsdp_kwargs["device_id"] = torch.cuda.current_device()
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        return FSDP(doctor_model, **fsdp_kwargs)
    return DDP(
        doctor_model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
    )


def save_doctor_model(doctor_model, tokenizer, save_dir: str, rank: int, distributed: bool):
    if is_fsdp_model(doctor_model):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(doctor_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = doctor_model.state_dict()
        if is_main_process(rank):
            os.makedirs(save_dir, exist_ok=True)
            state_dict = {
                (k[len("model."):] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }
            doctor_model.module.model.save_pretrained(save_dir, state_dict=state_dict)
            tokenizer.save_pretrained(save_dir)
    elif is_main_process(rank):
        os.makedirs(save_dir, exist_ok=True)
        unwrap_model(doctor_model).model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    if distributed:
        dist.barrier()


class DoctorAgentWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        *,
        inputs_embeds=None,
        input_ids=None,
        refined_context=None,
        attention_mask=None,
        **kwargs,
    ):
        if refined_context is not None:
            if input_ids is None:
                raise ValueError("input_ids are required when refined_context is provided.")
            prompt_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([refined_context, prompt_embeds], dim=1)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)


def build_epoch_shard(samples: List[Dict], epoch: int, rank: int, world_size: int, seed: int) -> List[Dict]:
    epoch_samples = list(samples)
    if not epoch_samples:
        return []
    random.Random(seed + epoch).shuffle(epoch_samples)

    if world_size == 1:
        return epoch_samples

    per_rank = math.ceil(len(epoch_samples) / world_size)
    target_len = per_rank * world_size
    original_len = len(epoch_samples)
    while len(epoch_samples) < target_len:
        epoch_samples.append(epoch_samples[(len(epoch_samples) - original_len) % original_len])
    return epoch_samples[rank:target_len:world_size]


def cleanup_distributed(distributed: bool):
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def sync_module_grads(modules, distributed: bool, world_size: int):
    if not distributed:
        return
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def get_answer_letter(sample: Dict) -> str:
    return sample.get("answer_idx", sample.get("answer", "A")).strip()[0].upper()


def get_full_context(sample: Dict, sep: str = " ") -> str:
    """
    Safely normalise the context field into a single string.
    """
    ctx = sample.get("context", "")
    if isinstance(ctx, list):
        parts = [s.strip() for s in ctx if isinstance(s, str) and s.strip()]
        return sep.join(parts)
    elif isinstance(ctx, str):
        return ctx.strip()
    else:
        return ""


def build_answer_prompt(sample: Dict, patient_info: str) -> str:
    options_text = "\n".join([f"{key}: {value}" for key, value in sample["options"].items()])
    return (
        f"Patient initial information:\n{patient_info}\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        "Select one correct answer from A to D. Answer with only the letter."
    )


def retrieve_knowledge_triplets(retriever: TripletRetriever | None, query: str, top_k: int):
    if retriever is None:
        return []
    return retriever.retrieve(query, top_k=top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/all_train_convo.jsonl")
    parser.add_argument("--expert_model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--triplet_model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--distributed_backend", choices=["ddp", "fsdp"], default="fsdp")
    parser.add_argument("--triplet_use_api", action="store_true", help="Use an OpenAI-compatible API for the frozen triplet agent")
    parser.add_argument("--triplet_use_vllm", action="store_true", help="Use the local vLLM-compatible path for the frozen triplet agent")

    parser.add_argument("--prefix_len", type=int, default=20)
    parser.add_argument("--refinement_steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_ckpt", default="save_model/graphmed_lt.ckpt")
    parser.add_argument("--save_doctor_dir", default="save_model/doctor_agent")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="print extracted triplets for inspection")
    parser.add_argument("--seed", type=int, default=42)

    # GNN dimensions
    parser.add_argument("--gnn_model", default="gat", choices=sorted(load_gnn_model))
    parser.add_argument("--gnn_in_dim", type=int, default=256)
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--triplet_corpus", default=None, help="Local external triplet corpus, e.g. a PrimeKG-derived triplet file")
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--max_corpus_triplets", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    if args.distributed_backend == "fsdp" and distributed and FSDP is None:
        raise RuntimeError("--distributed_backend fsdp was requested, but FSDP is unavailable.")
    if args.triplet_use_api and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")):
        raise RuntimeError("--triplet_use_api requires OPENAI_API_KEY or OPENAI_BASE_URL to be set.")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    local_accum_steps = max(1, math.ceil(args.batch_size / world_size))
    effective_batch_size = local_accum_steps * world_size

    if args.use_wandb and is_main_process(rank):
        wandb.init(project="Train_GraphMed_LT", name="full_finetune_latent_refinement", config=vars(args))

    samples = load_jsonl(args.train_file)
    if is_main_process(rank):
        logging.info(f"Loaded {len(samples)} samples from {args.train_file}")
        logging.info(
            f"Distributed={distributed} world_size={world_size} "
            f"local_accum_steps={local_accum_steps} effective_batch_size={effective_batch_size}"
        )

    # ----- Doctor LLM (full fine-tuning) -----
    train_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    load_device = "cpu" if distributed and args.distributed_backend == "fsdp" else train_device
    cache = helper.ModelCache(args.expert_model, max_tokens=512, device=load_device)
    tokenizer = cache.tokenizer
    raw_doctor_model = cache.model.train()
    if not (distributed and args.distributed_backend == "fsdp"):
        raw_doctor_model = raw_doctor_model.to(train_device)
    device = train_device
    if args.gradient_checkpointing and hasattr(raw_doctor_model, "gradient_checkpointing_enable"):
        raw_doctor_model.gradient_checkpointing_enable()
    for p in raw_doctor_model.parameters():
        p.requires_grad_(True)
    doctor_model = DoctorAgentWrapper(raw_doctor_model)
    doctor_model = wrap_distributed_doctor(
        doctor_model,
        raw_doctor_model,
        distributed=distributed,
        backend=args.distributed_backend,
        local_rank=local_rank,
    )
    hidden_size = raw_doctor_model.config.hidden_size

    # ----- BiCA (frozen) & text mappers (trainable) -----
    bica_model, bica_tokenizer, bica_device, bica_dim = init_bica()
    node_in, edge_in, source_in = make_text_mappers(
        embedding_dim=bica_dim,
        gnn_in_dim=args.gnn_in_dim,
        device=device,
        include_source=True,
    )

    retriever = None
    if args.triplet_corpus:
        retriever = TripletRetriever(
            args.triplet_corpus,
            encoder_model=bica_model,
            encoder_tokenizer=bica_tokenizer,
            encoder_device=bica_device,
            device=device,
            max_triplets=args.max_corpus_triplets,
        )
        if is_main_process(rank):
            logging.info(f"Loaded {len(retriever.records)} external triplets from {args.triplet_corpus}")

    # ----- Source-aware graph encoder -----
    graph_encoder = load_gnn_model[args.gnn_model](
        in_channels=args.gnn_in_dim,
        hidden_channels=args.gnn_hidden_dim,
        out_channels=args.gnn_hidden_dim,
        num_layers=args.gnn_layers,
        dropout=0.1,
        num_heads=args.gat_heads,
    ).to(device)

    # ----- Graph-to-token projector and latent thought refinement -----
    projector = TripletProjector(
        graph_encoder=graph_encoder,
        gnn_hidden_dim=args.gnn_hidden_dim,
        prefix_len=args.prefix_len,
        hidden_size=hidden_size
    ).to(device)
    if distributed:
        projector = DDP(
            projector,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
        )
    latent_refiner = LatentClinicalThoughtRefiner(
        doctor_model=doctor_model,
        refinement_steps=args.refinement_steps,
    )

    # ----- Optimiser: graph encoder + projector + source-aware mappers + doctor LLM -----
    optim_params = list(projector.parameters()) \
                   + list(node_in.parameters()) + list(edge_in.parameters()) \
                   + list(source_in.parameters()) + list(doctor_model.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    all_letters = None
    global_step = 0
    run_loss = 0.0

    for ep in range(args.epochs):
        rank_samples = build_epoch_shard(samples, ep, rank, world_size, args.seed)

        for local_step, sample in enumerate(
            tqdm(rank_samples, desc=f"Epoch {ep}", disable=not is_main_process(rank)),
            start=1,
        ):
            question = sample["question"]
            patient_info = sample.get("initial_info") or get_full_context(sample)

            # ----- Triplet extraction -----
            extracted_triplets = extract_triplets(
                patient_info=patient_info,
                question=question,
                qa_pairs=[],
                model_args={
                    "model_name": args.triplet_model,
                    "use_api": args.triplet_use_api,
                    "use_vllm": args.triplet_use_vllm,
                    "client": client if args.triplet_use_api else None,
                    "debug": False
                },
                choices=sample.get("options", None)
            )
            retrieved_triplets = retrieve_knowledge_triplets(
                retriever,
                query=patient_info,
                top_k=args.retrieval_top_k,
            )

            graph_memory = PatientGraphMemory()
            graph_memory.initialise(extracted_triplets)
            graph_memory.update([], retrieved_triplets)
            graph_triplets = graph_memory.records

            if args.debug:
                print("\n=================== Example Interaction ===================")
                print(f"Patient Info     : {patient_info}")
                print(f"Question         : {question}")
                print("Graph Memory Triplets:")
                for t in graph_memory.as_text_list(include_source=True):
                    print(f"  - {t}")

            # ----- Build prompt -----
            prompt = build_answer_prompt(sample, patient_info)

            # ----- Token embeddings for prompt -----
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # ----- Triplets to source-aware PyG graph (using BiCA externally) -----
            graph_data = triplets_to_graph(
                triplets=graph_triplets,
                encoder_model=bica_model,
                encoder_tokenizer=bica_tokenizer,
                encoder_device=bica_device,
                node_in=node_in,
                edge_in=edge_in,
                source_in=source_in,
                gnn_in_dim=args.gnn_in_dim,
                device=device,
            )

            # ----- Project to evidence tokens, refine latent thoughts, and concatenate -----
            evidence_tokens = projector(graph_data)  # (1, m, hidden)
            evidence_tokens = evidence_tokens.to(dtype=next(doctor_model.parameters()).dtype, device=device)
            refined_context = latent_refiner(evidence_tokens)  # (1, 2m, hidden)
            attn_mask = torch.ones(
                1,
                refined_context.size(1) + ids.size(1),
                dtype=torch.long,
                device=device,
            )

            out = doctor_model(
                input_ids=ids,
                refined_context=refined_context,
                attention_mask=attn_mask,
                use_cache=False,
            )

            # ----- Prepare target indices over answer letters -----
            if all_letters is None:
                # Expecting keys like {'A': '...', 'B': '...', ...}
                all_letters = sorted(sample["options"])

            target_ids = torch.tensor(
                [tokenizer.encode(l, add_special_tokens=False)[-1] for l in all_letters],
                device=device,
            )
            target_label = all_letters.index(get_answer_letter(sample))

            # ----- Loss: Cross-Entropy -----
            ce_loss = torch.nn.functional.cross_entropy(
                out.logits[:, -1, target_ids],
                torch.tensor([target_label], device=device)
            )
            (ce_loss / local_accum_steps).backward()

            global_step += 1
            run_loss += ce_loss.item()

            if is_main_process(rank):
                print(f"[Rank {rank} | Step {global_step}] CE: {ce_loss.item():.4f}")
            if args.use_wandb and is_main_process(rank):
                wandb.log({
                    "ce_loss": ce_loss.item(),
                    "loss": ce_loss.item(),
                    "step": global_step,
                    "epoch": ep,
                })

            if local_step % local_accum_steps == 0:
                sync_module_grads([node_in, edge_in, source_in], distributed, world_size)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if len(rank_samples) % local_accum_steps != 0:
            sync_module_grads([node_in, edge_in, source_in], distributed, world_size)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        mean_loss = run_loss / max(global_step, 1)
        if distributed:
            loss_state = torch.tensor([run_loss, float(global_step)], dtype=torch.float64, device=device)
            dist.all_reduce(loss_state, op=dist.ReduceOp.SUM)
            mean_loss = (loss_state[0] / loss_state[1].clamp_min(1.0)).item()
        if is_main_process(rank):
            logging.info(f"Epoch {ep} complete. Mean loss: {mean_loss:.4f}")
        if args.use_wandb and is_main_process(rank):
            wandb.log({"epoch_avg_loss": mean_loss, "epoch": ep})

        # ----- Save graph encoder + projector + mappers -----
        if is_main_process(rank):
            ckpt_dir = os.path.dirname(args.save_ckpt)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            base_projector = unwrap_model(projector)
            ckpt = {
                "projector": base_projector.state_dict(),
                "graph_encoder": base_projector.graph_encoder.state_dict(),
                "node_in": unwrap_model(node_in).state_dict(),
                "edge_in": unwrap_model(edge_in).state_dict(),
                "source_in": unwrap_model(source_in).state_dict(),
                "config": vars(args),
                "hidden_size": hidden_size,
            }
            torch.save(ckpt, args.save_ckpt)

    save_doctor_model(doctor_model, tokenizer, args.save_doctor_dir, rank, distributed)
    if is_main_process(rank):
        print("Training finished.")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
