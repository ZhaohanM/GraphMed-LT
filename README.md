# GraphMed-LT

**Patient-Specific Graph Memory with Latent Clinical Thought Refinement for Multi-Turn Medical Conversations**

## Overview

GraphMed-LT implements the system described in the paper. It converts patient responses into patient-specific clinical triplets, optionally retrieves relevant external knowledge triplets from a local PrimeKG-derived corpus, builds a source-aware patient graph memory, projects the graph into evidence tokens, and refines those tokens inside a trainable doctor agent through latent clinical thought refinement.

<p align="center">
  <img src="image/GraphMed-LT.png">
</p>

## Components

- **Patient agent**: frozen simulator that answers follow-up questions using only dataset-supported patient facts.
- **Triplet agent**: frozen extractor that creates observed patient triplets from the initial patient description and later patient responses.
- **External triplet retrieval**: optional top-3 retrieval from a local triplet corpus. In the paper this corpus is PrimeKG; this repository expects you to provide the local PrimeKG-derived triplet file through `--triplet_corpus`.
- **Patient-specific graph memory**: initializes `G_0` from the initial patient description `p_0`, then updates the graph after each patient response.
- **Source-aware graph encoder**: distinguishes patient-observed triplets from retrieved background triplets through source-type embeddings.
- **Graph-to-token projector**: maps the graph representation into `m=20` graph-conditioned evidence tokens.
- **Latent clinical thought refinement**: pairs each evidence token with a latent thought token and updates the thought tokens for `K=5` refinement steps inside the doctor agent.
- **Doctor agent**: trained with full fine-tuning together with the graph encoder and projector. The patient and triplet agents remain frozen.

## Installation

```bash
conda env create -f environment.yml
conda activate GraphMed-LT
```

## Training

The main training entry point is `projection_train.py`. Despite the historical filename, it now trains the graph encoder, graph-to-token projector, source-aware edge embeddings, latent refinement path, and doctor-agent parameters.

```bash
python projection_train.py \
  --train_file data/all_train_convo.jsonl \
  --expert_model meta-llama/Llama-3.1-8B-Instruct \
  --triplet_model llama-3.3-70b-instruct-awq \
  --triplet_corpus path/to/primekg_triplets.jsonl \
  --retrieval_top_k 3 \
  --prefix_len 20 \
  --refinement_steps 5 \
  --gnn_model gat \
  --gnn_in_dim 256 \
  --gnn_hidden_dim 256 \
  --gat_heads 4 \
  --lr 1e-5 \
  --batch_size 128 \
  --epochs 50
```

If `--triplet_corpus` is omitted, no retrieved background triplets are added. The code does not fabricate PrimeKG triples.

## Benchmark

```bash
python GraphMedLT_benchmark.py \
  --expert_module expert --expert_class ScaleExpert \
  --patient_module patient --patient_class FactSelectPatient \
  --data_dir data --dev_filename all_dev_good.jsonl \
  --projection_ckpt save_model/graphmed_lt.ckpt \
  --triplet_corpus path/to/primekg_triplets.jsonl \
  --output_filename results/graphmed_lt_dev.jsonl \
  --max_questions 10
```

The existing prompt-based expert classes keep a text fallback for API models. Embedding-level graph evidence and latent refinement are used directly in local training; API-only models cannot consume hidden-state evidence tokens.

## License

This repository is released under the MIT License.
