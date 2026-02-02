# TriMediQ  
**Triplet-Structured Knowledge Integration for Interactive Medical Question Answering**

## 📌 Overview

**TriMediQ** is a triplet-structured approach for **interactive medical question answering (QA)**, designed to support reliable **multi-turn clinical reasoning**.

- **Patient-specific knowledge graph construction**  
  Extracted triplets are incrementally assembled into a dynamic, patient-specific graph that evolves during interaction.

- **Trainable projection module**  
  A graph encoder + projector injects structured graph representations into a frozen doctor LLM via prefix-style embedding-level integration.```

---

## 🧩 Framework Overview

<p align="center">
  <img src="image/TriMediQ.png">
</p>

**TriMediQ consists of three agents and one projection module:**

- **Doctor Agent**  
  A frozen LLM that performs structured multi-hop reasoning over injected graph representations to decide whether to ask follow-up questions or output a final diagnosis.

- **Patient Agent**  
  A frozen LLM that responds to doctor questions using dataset-curated patient records.

- **Triplet Agent**  
  A frozen LLM that extracts structured clinical triplets from patient responses.

- **Projection Module**  
  - **Graph Encoder** (GAT-based): encodes the patient-specific knowledge graph  
  - **Projector**: aligns graph representations with the doctor LLM embedding space

---

## ⚙️ Installation

Create a new conda environment (GPU + CUDA required for PyTorch):

```bash
conda env create -f environment.yml
conda activate TriMediQ
```

## ▶️ Running the Benchmark
Example run:
```bash
python TriMediQ_benchmark.py \
  --expert_module expert --expert_class ScaleExpert \
  --patient_module patient --patient_class FactSelectPatient \
  --data_dir ../data --dev_filename all_dev_good.jsonl \
  --output_filename out.jsonl --max_questions 10
```

## 📜 License
This repository is released under the MIT License.
