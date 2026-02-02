from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data
from utils.lm_modeling import load_model, load_text2embedding


# -------------------------------
# SBERT initialisation & helpers
# -------------------------------
def init_sbert() -> tuple[torch.nn.Module, object, torch.device, int]:
    """
    Initialise SBERT via your lm_modeling utilities.
    Returns:
        sbert_model, sbert_tokenizer, sbert_device, sbert_dim
    """
    sbert_model, sbert_tokenizer, sbert_device = load_model['sbert']()
    dummy = sber_text2embedding(sbert_model, sbert_tokenizer, sbert_device, ["_"])
    sbert_dim = int(dummy.shape[-1]) if dummy.numel() > 0 else 1024  # all-roberta-large-v1  1024
    return sbert_model, sbert_tokenizer, sbert_device, sbert_dim


def sber_text2embedding(
    sbert_model,
    sbert_tokenizer,
    sbert_device: torch.device,
    texts: List[str],
) -> torch.Tensor:
    """
    Wrap lm_modeling.load_text2embedding['sbert'] for clarity.
    Returns: (N, sbert_dim) tensor on CPU (与原实现一致)
    """
    return load_text2embedding['sbert'](sbert_model, sbert_tokenizer, sbert_device, texts)


# ----------------------------------------
# Trainable mappers: SBERT -> GNN(in_dim)
# ----------------------------------------
def make_text_mappers(
    sbert_dim: int,
    gnn_in_dim: int,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[nn.Linear, nn.Linear]:
    """
    Create trainable linear maps to project SBERT embeddings into the GNN input space.
    Returns:
        node_in: Linear(sbert_dim -> gnn_in_dim)
        edge_in: Linear(sbert_dim -> gnn_in_dim)
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_in = nn.Linear(sbert_dim, gnn_in_dim, bias=False).to(dev)
    edge_in = nn.Linear(sbert_dim, gnn_in_dim, bias=False).to(dev)
    return node_in, edge_in


# -----------------------------
# Triplets → PyG graph (Data)
# -----------------------------
def _parse_triplets(
    triplets: List[Union[str, Tuple[str, str, str]]]
) -> tuple[List[str], List[Tuple[int, int]], List[str]]:
    """
    Normalise triplets to nodes/edges/rels.
    Supports:
        • List[Tuple[head, rel, tail]]
        • List[str] where each is "head; relation; tail"
    Returns:
        nodes: List[str], edges: List[(int,int)], rels: List[str]
    """
    node2id: Dict[str, int] = {}
    edges: List[Tuple[int, int]] = []
    rels: List[str] = []

    for t in triplets:
        if isinstance(t, (list, tuple)) and len(t) == 3:
            h, r, o = (str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip())
        elif isinstance(t, str) and ";" in t:
            parts = [p.strip() for p in t.split(";")]
            if len(parts) < 3:
                continue
            h, r, o = parts[0], parts[1], parts[2]
        else:
            continue

        for n in (h, o):
            if n not in node2id:
                node2id[n] = len(node2id)
        edges.append((node2id[h], node2id[o]))
        rels.append(r)

    nodes: List[str] = [None] * len(node2id)  # type: ignore[assignment]
    for n, i in node2id.items():
        nodes[i] = n
    return nodes, edges, rels


def triplets_to_graph(
    triplets: List[Union[str, Tuple[str, str, str]]],
    *,
    sbert_model,
    sbert_tokenizer,
    sbert_device: torch.device,
    node_in: nn.Linear,
    edge_in: nn.Linear,
    gnn_in_dim: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Data:
    """
    Build a PyG Data graph from triplets, using SBERT to embed nodes/relations,
    then mapping them to GNN input dimension via (trainable) linear layers.

    Args:
        triplets: list of "h;r;t" or (h, r, t)
        sbert_model/tokenizer/device: from init_sbert()
        node_in/edge_in: trainable Linear layers from make_text_mappers()
        gnn_in_dim: input dim expected by your GNN (in_channels)
        device: target device for the resulting tensors

    Returns:
        Data with:
            x:         (num_nodes, gnn_in_dim)
            edge_index:(2, num_edges)
            edge_attr: (num_edges, gnn_in_dim)
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes, edges, rels = _parse_triplets(triplets)

    if len(nodes) == 0:
        x = torch.zeros(1, gnn_in_dim, device=dev)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=dev)
        edge_attr = torch.zeros(0, gnn_in_dim, device=dev)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=1)

    with torch.no_grad():
        node_vecs = sber_text2embedding(sbert_model, sbert_tokenizer, sbert_device, nodes)  # (N, sbert_dim) on CPU
        rel_vecs  = sber_text2embedding(sbert_model, sbert_tokenizer, sbert_device, rels)   # (E, sbert_dim) on CPU

    x = node_in(node_vecs.to(node_in.weight.device)).to(dev)  # (N, gnn_in_dim)
    e = edge_in(rel_vecs.to(edge_in.weight.device)).to(dev)   # (E, gnn_in_dim)

    edge_index = torch.tensor(edges, dtype=torch.long, device=dev).t().contiguous()
    return Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=x.shape[0])


__all__ = [
    "init_sbert",
    "make_text_mappers",
    "triplets_to_graph",
    "sber_text2embedding",
]
