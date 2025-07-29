import torch
from torch import Tensor, nn
from .embedder import Embedder
from .gnn_layer import GNNLayer
from .out_layer import OutLayer
import torch.nn.functional as F

class GCNEncoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        gcn_layer_num: int,
        out_layer_num: int,
    ):
        super(GCNEncoder, self).__init__()
        self.embed = Embedder(hidden_dim)
        self.gcn_layers = nn.ModuleList(
            [GNNLayer(hidden_dim) for _ in range(gcn_layer_num)]
        )
        self.out = OutLayer(hidden_dim, out_layer_num)
        
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor):
        """
        Args:
            x: (B, V, 2) nodes_feature (node coords)
            e: (B, E) edges_feature (distance matrix)
            edge_index: (B, 2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            prob: (B, E, 2) Probability of each edge being connected and not connected to the TSP tour.
        """
        batch_size = x.shape[0]
        e_out = []
        for idx in range(batch_size):
            x_i, e_i = x[idx], e[idx]
            x_i, e_i = self.embed(x_i, e_i)
            for gcn_layer in self.gcn_layers:
                x_i, e_i = gcn_layer(x_i, e_i, edge_index[idx])
            e_i = self.out(e_i)
            e_i = F.sigmoid(e_i)
            e_out.append(e_i)
        e_out = torch.stack(e_out, dim=0)  # shape: (B, E, 2)
        return e_out