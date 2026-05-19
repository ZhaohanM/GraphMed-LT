from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data


class TripletProjector(nn.Module):
    """
    Project a patient-specific graph memory into graph-conditioned evidence tokens.

    Inputs:
      • graph_encoder: a GNN (e.g., GCN/GAT/GraphTransformer) whose forward returns
          node_embeds, edge_attr = graph_encoder(x, edge_index, edge_attr)
        where node_embeds has shape (num_nodes, gnn_hidden_dim)
      • gnn_hidden_dim: output hidden size of the graph_encoder
      • prefix_len: number of graph-conditioned evidence tokens
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

        self.pool_gate = nn.Linear(gnn_hidden_dim, 1)

        # GNN graph embedding → m distinct LLM evidence tokens
        self.projector = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, prefix_len * hidden_size),
        )

    def _pool_graph(self, node_embeds: torch.Tensor, graph: Data) -> torch.Tensor:
        """
        Attention-pool node embeddings to obtain a single graph embedding.
        Supports both single-graph and batched-graph inputs.
        """
        if node_embeds.numel() == 0:
            return torch.zeros(self.gnn_hidden_dim, device=node_embeds.device, dtype=node_embeds.dtype)

        if hasattr(graph, "batch") and graph.batch is not None:
            batch = graph.batch.to(node_embeds.device)
            graph_embeds = []
            for graph_id in torch.unique(batch, sorted=True):
                mask = batch == graph_id
                scores = self.pool_gate(node_embeds[mask]).squeeze(-1)
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                graph_embeds.append(torch.sum(weights * node_embeds[mask], dim=0))
            return torch.stack(graph_embeds, dim=0)[-1]

        scores = self.pool_gate(node_embeds).squeeze(-1)
        weights = torch.softmax(scores, dim=0).unsqueeze(-1)
        return torch.sum(weights * node_embeds, dim=0)

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

        projected = self.projector(graph_embed)                        # (prefix_len * hidden_size,)
        prefix = projected.view(self.prefix_len, self.hidden_size)      # (prefix_len, hidden_size)
        return prefix.unsqueeze(0)                                     # (1, prefix_len, hidden_size)
