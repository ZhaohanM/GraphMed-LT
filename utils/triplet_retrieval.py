from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from utils.graph_memory import RETRIEVED_SOURCE, normalize_triplet_record, triplet_to_text
from utils.triplets_to_graph import bica_text2embedding


def _load_json_or_jsonl(path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            payload = json.load(f)
            rows = payload if isinstance(payload, list) else payload.get("triplets", [])
            for row in rows:
                record = normalize_triplet_record(row, default_source=RETRIEVED_SOURCE)
                if record is not None:
                    records.append(record)
            return records

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = line
            record = normalize_triplet_record(row, default_source=RETRIEVED_SOURCE)
            if record is not None:
                records.append(record)
    return records


def _load_delimited(path: str, delimiter: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames and any(name in reader.fieldnames for name in ("head", "subject", "h")):
            for row in reader:
                record = normalize_triplet_record(row, default_source=RETRIEVED_SOURCE)
                if record is not None:
                    records.append(record)
            return records

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            record = normalize_triplet_record(tuple(row[:3]), default_source=RETRIEVED_SOURCE)
            if record is not None:
                records.append(record)
    return records


def load_triplet_corpus(path: str, max_triplets: Optional[int] = None) -> List[Dict[str, str]]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Triplet corpus not found: {path}")

    lower = path.lower()
    if lower.endswith((".jsonl", ".json")):
        records = _load_json_or_jsonl(path)
    elif lower.endswith(".tsv"):
        records = _load_delimited(path, "\t")
    elif lower.endswith(".csv"):
        records = _load_delimited(path, ",")
    else:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = normalize_triplet_record(line, default_source=RETRIEVED_SOURCE)
                if record is not None:
                    records.append(record)

    if max_triplets is not None:
        records = records[:max_triplets]
    return records


class TripletRetriever:
    """BiCA similarity retriever for a local PrimeKG-derived triplet corpus."""

    def __init__(
        self,
        corpus_path: str,
        *,
        encoder_model=None,
        encoder_tokenizer=None,
        encoder_device=None,
        sbert_model=None,
        sbert_tokenizer=None,
        sbert_device=None,
        device: torch.device,
        max_triplets: Optional[int] = None,
    ) -> None:
        self.records = load_triplet_corpus(corpus_path, max_triplets=max_triplets)
        self.device = device
        self.encoder_model = encoder_model if encoder_model is not None else sbert_model
        self.encoder_tokenizer = encoder_tokenizer if encoder_tokenizer is not None else sbert_tokenizer
        self.encoder_device = encoder_device if encoder_device is not None else sbert_device
        if self.encoder_model is None or self.encoder_device is None:
            raise ValueError("A BiCA encoder model and device are required")

        if self.records:
            texts = [triplet_to_text(record, include_source=False) for record in self.records]
            with torch.no_grad():
                self.embeddings = bica_text2embedding(
                    self.encoder_model,
                    self.encoder_tokenizer,
                    self.encoder_device,
                    texts,
                ).to(self.device)
                self.embeddings = F.normalize(self.embeddings, p=2, dim=-1)
        else:
            self.embeddings = torch.empty(0, device=self.device)

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not query or not self.records or top_k <= 0:
            return []

        query_embedding = bica_text2embedding(
            self.encoder_model,
            self.encoder_tokenizer,
            self.encoder_device,
            [query],
        ).to(self.device)
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)

        scores = query_embedding @ self.embeddings.T
        k = min(top_k, scores.shape[-1])
        top_indices = torch.topk(scores.squeeze(0), k=k).indices.tolist()
        return [dict(self.records[idx], source=RETRIEVED_SOURCE) for idx in top_indices]
