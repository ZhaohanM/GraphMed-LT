from __future__ import annotations
import json
import random
import logging
import torch
import argparse
from typing import List, Dict
from tqdm import tqdm

from utils.triplet_projector import TripletProjector
from triplet_extraction import extract_triplets
from utils.gnn import load_gnn_model
from utils.triplets_to_graph import init_sbert, make_text_mappers, triplets_to_graph
import helper
import wandb

from openai import OpenAI


# NOTE: Replace api_key with your own credential or rely on environment variables.
client = OpenAI(
    base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1",
    api_key="YOUR_API_KEY_HERE"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/all_train_convo.jsonl")
    parser.add_argument("--expert_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--triplet_model", default="llama-3.3-70b-instruct-awq")

    parser.add_argument("--prefix_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_ckpt", default="save_model/projection.ckpt")
    parser.add_argument("--debug", action="store_true", help="print extracted triplets for inspection")

    # GNN dimensions
    parser.add_argument("--gnn_in_dim", type=int, default=768)
    parser.add_argument("--gnn_hidden_dim", type=int, default=768)

    args = parser.parse_args()

    wandb.init(project="Train_Projection_Module_of_TriMediQ", name="proj_ce_only_run", config=vars(args))

    samples = load_jsonl(args.train_file)
    logging.info(f"Loaded {len(samples)} samples from {args.train_file}")

    # ----- Expert LLM (frozen) -----
    cache = helper.ModelCache(args.expert_model, max_tokens=512)
    tokenizer = cache.tokenizer
    model = cache.model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ----- SBERT (frozen) & text mappers (trainable) -----
    sbert_model, sbert_tokenizer, sbert_device, sbert_dim = init_sbert()
    node_in, edge_in = make_text_mappers(
        sbert_dim=sbert_dim,
        gnn_in_dim=args.gnn_in_dim,
        device=model.device
    )

    # ----- GNN encoder (trainable unless you choose to freeze) -----
    graph_encoder = load_gnn_model["gcn"](
        in_channels=args.gnn_in_dim,
        hidden_channels=args.gnn_hidden_dim,
        out_channels=args.gnn_hidden_dim,
        num_layers=2,
        dropout=0.1,
    ).cuda()

    # ----- Triplet projector (GNN → LLM soft prefix) -----
    projector = TripletProjector(
        graph_encoder=graph_encoder,
        gnn_hidden_dim=args.gnn_hidden_dim,
        prefix_len=args.prefix_len,
        hidden_size=model.config.hidden_size
    ).cuda()

    # ----- Optimiser: projector + graph_encoder + (node_in/edge_in) -----
    optim_params = list(projector.parameters()) + list(graph_encoder.parameters()) \
                   + list(node_in.parameters()) + list(edge_in.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)

    all_letters = None
    global_step = 0
    run_loss = 0.0

    for ep in range(args.epochs):
        random.shuffle(samples)

        for sample in tqdm(samples, desc=f"Epoch {ep}"):
            question = sample["question"]
            patient_info = get_full_context(sample)

            # ----- Triplet extraction -----
            triplets = extract_triplets(
                patient_info=patient_info,
                question=question,
                qa_pairs=[("Follow-up", sample.get("patient_response", ""))],
                model_args={
                    "model_name": args.triplet_model,
                    "use_api": True,
                    "client": client,
                    "debug": False
                },
                choices=sample.get("options", None)
            )

            if args.debug:
                print("\n=================== Example Interaction ===================")
                print(f"Patient Info     : {patient_info}")
                print(f"Question         : {question}")
                print("Extracted Triplets:")
                for t in triplets:
                    print(f"  - {t}")

            # ----- Build prompt -----
            options_text = ", ".join([f"{k}: {v}" for k, v in sample["options"].items()])
            prompt = (
                f"Question: {question}\n"
                f"Initial Info: {patient_info}\n"
                f"Options: {options_text}"
            )

            # ----- Token embeddings for prompt -----
            ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            txt_emb = model.get_input_embeddings()(ids)

            # ----- Triplets → PyG Graph (using SBERT externally) -----
            graph_data = triplets_to_graph(
                triplets=triplets,
                sbert_model=sbert_model,
                sbert_tokenizer=sbert_tokenizer,
                sbert_device=sbert_device,
                node_in=node_in,
                edge_in=edge_in,
                gnn_in_dim=args.gnn_in_dim,
                device=txt_emb.device,
            )

            # ----- Project to soft prefix and concatenate -----
            prefix_emb = projector(graph_data)  # (1, prefix_len, hidden)
            prefix_emb = prefix_emb.to(dtype=txt_emb.dtype, device=txt_emb.device)
            attn_mask = torch.ones(1, prefix_emb.size(1) + txt_emb.size(1), dtype=torch.long).cuda()

            out = model(
                inputs_embeds=torch.cat([prefix_emb, txt_emb], dim=1),
                attention_mask=attn_mask
            )

            # ----- Prepare target indices over answer letters -----
            if all_letters is None:
                # Expecting keys like {'A': '...', 'B': '...', ...}
                all_letters = sorted(sample["options"])

            target_ids = torch.tensor(
                [tokenizer.convert_tokens_to_ids(l) for l in all_letters]
            ).cuda()
            target_label = all_letters.index(get_answer_letter(sample))

            # ----- Loss: Cross-Entropy -----
            ce_loss = torch.nn.functional.cross_entropy(
                out.logits[:, -1, target_ids],
                torch.tensor([target_label]).cuda()
            )
            ce_loss.backward()

            global_step += 1
            run_loss += ce_loss.item()

            print(f"[Step {global_step}] CE: {ce_loss.item():.4f}")
            wandb.log({
                "ce_loss": ce_loss.item(),
                "loss": ce_loss.item(),
                "step": global_step,
                "epoch": ep,
            })

            if global_step % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        mean_loss = run_loss / max(global_step, 1)
        logging.info(f"Epoch {ep} complete. Mean loss: {mean_loss:.4f}")
        wandb.log({"epoch_avg_loss": mean_loss, "epoch": ep})

        # ----- Save projector + GNN + mappers -----
        ckpt = {
            "projector": projector.state_dict(),
            "graph_encoder": graph_encoder.state_dict(),
            "node_in": node_in.state_dict(),
            "edge_in": edge_in.state_dict(),
            "config": vars(args),
        }
        torch.save(ckpt, args.save_ckpt)

    print("Training finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
