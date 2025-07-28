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
        gcn_layers = [GNNLayer(hidden_dim) for _ in range(gcn_layer_num)]
        self.gcn = nn.Sequential(*gcn_layers)
        self.out = OutLayer(hidden_dim, out_layer_num)
        
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor):
        """
        Args:
            x: (V, 2) nodes_feature (node coords)
            e: (E,) edges_feature (distance matrix)
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            prob: (E, ) probablity of each edge being connected to the TSP tour
        """
        x, e = self.embed(x, e)
        x, e = self.gcn(x, e, edge_index) 
        e = self.out(e)
        return F.sigmoid(e)  # shape: (E, )