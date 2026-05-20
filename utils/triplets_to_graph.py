from __future__ import annotations
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data
from utils.lm_modeling import load_model, load_text2embedding
from utils.graph_memory import PATIENT_SOURCE, normalize_triplet_record


def init_bica() -> tuple[torch.nn.Module, object, torch.device, int]:
    """Initialise the frozen BiCA encoder used for retrieval and graph nodes."""
    encoder_model, encoder_tokenizer, encoder_device = load_model["bica"]()
    dummy = bica_text2embedding(encoder_model, encoder_tokenizer, encoder_device, ["_"])
    encoder_dim = int(dummy.shape[-1]) if dummy.numel() > 0 else 768
    return encoder_model, encoder_tokenizer, encoder_device, encoder_dim


def bica_text2embedding(
    encoder_model,
    encoder_tokenizer,
    encoder_device: torch.device,
    texts: List[str],
) -> torch.Tensor:
    """Return BiCA embeddings on CPU, matching the graph-building path."""
    return load_text2embedding["bica"](encoder_model, encoder_tokenizer, encoder_device, texts)


def init_sbert() -> tuple[torch.nn.Module, object, torch.device, int]:
    return init_bica()


def sber_text2embedding(encoder_model, encoder_tokenizer, encoder_device: torch.device, texts: List[str]) -> torch.Tensor:
    return bica_text2embedding(encoder_model, encoder_tokenizer, encoder_device, texts)


def make_text_mappers(
    embedding_dim: Optional[int] = None,
    gnn_in_dim: int = 256,
    device: Optional[Union[str, torch.device]] = None,
    include_source: bool = False,
    sbert_dim: Optional[int] = None,
):
    """Create trainable maps from BiCA embeddings into the GNN input space."""
    if embedding_dim is None:
        embedding_dim = sbert_dim
    if embedding_dim is None:
        raise ValueError("embedding_dim is required")

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_in = nn.Linear(embedding_dim, gnn_in_dim, bias=False).to(dev)
    edge_in = nn.Linear(embedding_dim, gnn_in_dim, bias=False).to(dev)
    if include_source:
        source_in = nn.Embedding(2, gnn_in_dim).to(dev)
        return node_in, edge_in, source_in
    return node_in, edge_in


def _parse_triplets(
    triplets: List[Union[str, Tuple[str, str, str], Tuple[str, str, str, str], dict]]
) -> tuple[List[str], List[Tuple[int, int]], List[str], List[str]]:
    node2id: dict[str, int] = {}
    edges: List[Tuple[int, int]] = []
    rels: List[str] = []
    sources: List[str] = []

    for t in triplets:
        record = normalize_triplet_record(t, default_source=PATIENT_SOURCE)
        if record is None:
            continue
        h, r, o = record["head"], record["relation"], record["tail"]

        for n in (h, o):
            if n not in node2id:
                node2id[n] = len(node2id)
        edges.append((node2id[h], node2id[o]))
        rels.append(r)
        sources.append(record.get("source", PATIENT_SOURCE))

    nodes: List[str] = [None] * len(node2id)  # type: ignore[assignment]
    for n, i in node2id.items():
        nodes[i] = n
    return nodes, edges, rels, sources


def triplets_to_graph(
    triplets: List[Union[str, Tuple[str, str, str], Tuple[str, str, str, str], dict]],
    *,
    node_in: nn.Linear,
    edge_in: nn.Linear,
    gnn_in_dim: int,
    encoder_model=None,
    encoder_tokenizer=None,
    encoder_device: Optional[torch.device] = None,
    sbert_model=None,
    sbert_tokenizer=None,
    sbert_device: Optional[torch.device] = None,
    source_in: Optional[nn.Embedding] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Data:
    """Build a source-aware PyG graph from patient and retrieved triplets."""
    encoder_model = encoder_model if encoder_model is not None else sbert_model
    encoder_tokenizer = encoder_tokenizer if encoder_tokenizer is not None else sbert_tokenizer
    encoder_device = encoder_device if encoder_device is not None else sbert_device
    if encoder_model is None or encoder_device is None:
        raise ValueError("A BiCA encoder model and device are required")

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_module = getattr(node_in, "module", node_in)
    edge_module = getattr(edge_in, "module", edge_in)
    source_module = getattr(source_in, "module", source_in) if source_in is not None else None

    nodes, edges, rels, sources = _parse_triplets(triplets)

    if len(nodes) == 0:
        x = torch.zeros(1, gnn_in_dim, device=dev)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=dev)
        edge_attr = torch.zeros(0, gnn_in_dim, device=dev)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=1)

    with torch.no_grad():
        node_vecs = bica_text2embedding(encoder_model, encoder_tokenizer, encoder_device, nodes)
        rel_vecs = bica_text2embedding(encoder_model, encoder_tokenizer, encoder_device, rels)

    x = node_in(node_vecs.to(node_module.weight.device)).to(dev)
    e = edge_in(rel_vecs.to(edge_module.weight.device)).to(dev)

    source_ids = torch.tensor(
        [0 if source == PATIENT_SOURCE else 1 for source in sources],
        dtype=torch.long,
        device=dev,
    )
    if source_in is not None and source_module is not None and source_ids.numel() > 0:
        e = e + source_in(source_ids.to(source_module.weight.device)).to(dev)

    edge_index = torch.tensor(edges, dtype=torch.long, device=dev).t().contiguous()
    return Data(x=x, edge_index=edge_index, edge_attr=e, source_type=source_ids, num_nodes=x.shape[0])


__all__ = [
    "init_bica",
    "bica_text2embedding",
    "init_sbert",
    "sber_text2embedding",
    "make_text_mappers",
    "triplets_to_graph",
]
