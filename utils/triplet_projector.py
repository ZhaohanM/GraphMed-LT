from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter


class TripletProjector(nn.Module):
    """
    Project a triplet graph embedding into a soft prefix for a frozen expert LLM.

    Inputs:
      • graph_encoder: a GNN (e.g., GCN/GAT/GraphTransformer) whose forward returns
          node_embeds, edge_attr = graph_encoder(x, edge_index, edge_attr)
        where node_embeds has shape (num_nodes, gnn_hidden_dim)
      • gnn_hidden_dim: output hidden size of the graph_encoder
      • prefix_len: number of soft-prefix tokens to prepend
      • hidden_size: embedding size of the target LLM (e.g., 4096 for LLaMA)

    Forward:
      • Accepts a PyG Data graph with fields:
          - x:         (num_nodes, gnn_in_dim)
          - edge_index:(2, num_edges)
          - edge_attr: (num_edges, gnn_in_dim)   # if your GNN uses edge features
          - batch:     (num_nodes,) optional, for batched graphs
      • Returns a tensor of shape (1, prefix_len, hidden_size)
    """

    def __init__(
        self,
        *,
        graph_encoder: nn.Module,
        gnn_hidden_dim: int,
        prefix_len: int = 20,
        hidden_size: int = 4096,
    ) -> None:
        super().__init__()
        self.graph_encoder = graph_encoder
        self.gnn_hidden_dim = gnn_hidden_dim
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size

        # GNN (gnn_hidden_dim) → LLM embedding (hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, hidden_size),
        )

    def _pool_graph(self, node_embeds: torch.Tensor, graph: Data) -> torch.Tensor:
        """
        Mean-pool node embeddings to obtain a single graph embedding.
        Supports both single-graph and batched-graph inputs.
        """
        if hasattr(graph, "batch") and graph.batch is not None:
            # Batched graphs: (num_nodes, hidden) → (batch_size, hidden)
            batch = graph.batch.to(node_embeds.device)
            graph_embeds = scatter(node_embeds, batch, dim=0, reduce="mean")
            # For this projector we expect a single graph; if batched, take the last graph in the batch.
            return graph_embeds[-1]
        # Single graph
        if node_embeds.numel() == 0:
            return torch.zeros(self.gnn_hidden_dim, device=node_embeds.device, dtype=node_embeds.dtype)
        return node_embeds.mean(dim=0)

    def forward(self, graph: Data) -> torch.Tensor:
        """
        graph: torch_geometric.data.Data
        Returns:
            prefix: (1, prefix_len, hidden_size)
        """
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = getattr(graph, "edge_attr", None)

        node_embeds, _ = self.graph_encoder(x, edge_index, edge_attr)  # (num_nodes, gnn_hidden_dim)
        graph_embed = self._pool_graph(node_embeds, graph)             # (gnn_hidden_dim,)

        projected = self.projector(graph_embed)                        # (hidden_size,)
        prefix = projected.unsqueeze(0).repeat(self.prefix_len, 1)     # (prefix_len, hidden_size)
        return prefix.unsqueeze(0)                                     # (1, prefix_len, hidden_size)
