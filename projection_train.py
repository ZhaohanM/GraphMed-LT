from __future__ import annotations
import json
import random
import logging
import torch
import argparse
import os
from typing import List, Dict
from tqdm import tqdm

from utils.triplet_projector import TripletProjector
from utils.latent_refinement import LatentClinicalThoughtRefiner
from utils.graph_memory import PatientGraphMemory
from utils.triplet_retrieval import TripletRetriever
from triplet_extraction import extract_triplets
from utils.gnn import load_gnn_model
from utils.triplets_to_graph import init_sbert, make_text_mappers, triplets_to_graph
import helper
import wandb

from openai import OpenAI


client = OpenAI(
    base_url=os.environ.get("IDA_LLM_BASE_URL", "http://api.llm.apps.os.dcs.gla.ac.uk/v1"),
    api_key=os.environ.get("IDA_LLM_API_KEY", "YOUR_API_KEY_HERE")
)


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
    parser.add_argument("--expert_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--triplet_model", default="llama-3.3-70b-instruct-awq")

    parser.add_argument("--prefix_len", type=int, default=20)
    parser.add_argument("--refinement_steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_ckpt", default="save_model/graphmed_lt.ckpt")
    parser.add_argument("--save_doctor_dir", default="save_model/doctor_agent")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="print extracted triplets for inspection")

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

    if args.use_wandb:
        wandb.init(project="Train_GraphMed_LT", name="full_finetune_latent_refinement", config=vars(args))

    samples = load_jsonl(args.train_file)
    logging.info(f"Loaded {len(samples)} samples from {args.train_file}")

    # ----- Doctor LLM (full fine-tuning) -----
    cache = helper.ModelCache(args.expert_model, max_tokens=512)
    tokenizer = cache.tokenizer
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cache.model.to(train_device).train()
    device = next(model.parameters()).device
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad_(True)

    # ----- SBERT (frozen) & text mappers (trainable) -----
    sbert_model, sbert_tokenizer, sbert_device, sbert_dim = init_sbert()
    node_in, edge_in, source_in = make_text_mappers(
        sbert_dim=sbert_dim,
        gnn_in_dim=args.gnn_in_dim,
        device=device,
        include_source=True,
    )

    retriever = None
    if args.triplet_corpus:
        retriever = TripletRetriever(
            args.triplet_corpus,
            sbert_model=sbert_model,
            sbert_tokenizer=sbert_tokenizer,
            sbert_device=sbert_device,
            device=device,
            max_triplets=args.max_corpus_triplets,
        )
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
        hidden_size=model.config.hidden_size
    ).to(device)
    latent_refiner = LatentClinicalThoughtRefiner(
        doctor_model=model,
        refinement_steps=args.refinement_steps,
    )

    # ----- Optimiser: graph encoder + projector + source-aware mappers + doctor LLM -----
    optim_params = list(projector.parameters()) + list(graph_encoder.parameters()) \
                   + list(node_in.parameters()) + list(edge_in.parameters()) \
                   + list(source_in.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)
    optimizer.zero_grad(set_to_none=True)

    all_letters = None
    global_step = 0
    run_loss = 0.0

    for ep in range(args.epochs):
        random.shuffle(samples)

        for sample in tqdm(samples, desc=f"Epoch {ep}"):
            question = sample["question"]
            patient_info = sample.get("initial_info") or get_full_context(sample)

            # ----- Triplet extraction -----
            extracted_triplets = extract_triplets(
                patient_info=patient_info,
                question=question,
                qa_pairs=[],
                model_args={
                    "model_name": args.triplet_model,
                    "use_api": True,
                    "client": client,
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
            txt_emb = model.get_input_embeddings()(ids)

            # ----- Triplets → PyG Graph (using SBERT externally) -----
            graph_data = triplets_to_graph(
                triplets=graph_triplets,
                sbert_model=sbert_model,
                sbert_tokenizer=sbert_tokenizer,
                sbert_device=sbert_device,
                node_in=node_in,
                edge_in=edge_in,
                source_in=source_in,
                gnn_in_dim=args.gnn_in_dim,
                device=txt_emb.device,
            )

            # ----- Project to evidence tokens, refine latent thoughts, and concatenate -----
            evidence_tokens = projector(graph_data)  # (1, m, hidden)
            evidence_tokens = evidence_tokens.to(dtype=txt_emb.dtype, device=txt_emb.device)
            refined_context = latent_refiner(evidence_tokens)  # (1, 2m, hidden)
            attn_mask = torch.ones(
                1,
                refined_context.size(1) + txt_emb.size(1),
                dtype=torch.long,
                device=device,
            )

            out = model(
                inputs_embeds=torch.cat([refined_context, txt_emb], dim=1),
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
            (ce_loss / args.batch_size).backward()

            global_step += 1
            run_loss += ce_loss.item()

            print(f"[Step {global_step}] CE: {ce_loss.item():.4f}")
            if args.use_wandb:
                wandb.log({
                    "ce_loss": ce_loss.item(),
                    "loss": ce_loss.item(),
                    "step": global_step,
                    "epoch": ep,
                })

            if global_step % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if global_step % args.batch_size != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        mean_loss = run_loss / max(global_step, 1)
        logging.info(f"Epoch {ep} complete. Mean loss: {mean_loss:.4f}")
        if args.use_wandb:
            wandb.log({"epoch_avg_loss": mean_loss, "epoch": ep})

        # ----- Save graph encoder + projector + mappers -----
        ckpt_dir = os.path.dirname(args.save_ckpt)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "projector": projector.state_dict(),
            "graph_encoder": graph_encoder.state_dict(),
            "node_in": node_in.state_dict(),
            "edge_in": edge_in.state_dict(),
            "source_in": source_in.state_dict(),
            "config": vars(args),
            "hidden_size": model.config.hidden_size,
        }
        torch.save(ckpt, args.save_ckpt)

    os.makedirs(args.save_doctor_dir, exist_ok=True)
    model.save_pretrained(args.save_doctor_dir)
    tokenizer.save_pretrained(args.save_doctor_dir)
    print("Training finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
