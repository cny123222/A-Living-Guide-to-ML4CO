from torch import Tensor, nn
import torch
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GNNLayer, self).__init__()
        
        # node updates
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # edge updates
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # BatchNorm for node and edge
        self.bn_x = nn.BatchNorm1d(hidden_dim)
        self.bn_e = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor):
        """
        Args:
            x: (V, H) Node features; e: (E, H) Edge features
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            Updated x and e after one layer of GNN.
        """
        # Deconstruct edge_index
        src, dest = edge_index  # shape: (E, )
        
        # --- Node Update ---
        w2_x_src = self.W2(x[src])  # shape: (E, H)
        messages = e * w2_x_src   # shape: (E, H)
        aggr_messages = torch.zeros_like(x)   # shape: (V, H)
        # index_add_ adds the 'messages' to 'aggr_messages' at indices specified by 'dest'
        aggr_messages.index_add_(0, dest, messages)   # shape: (V, H)
        x_new = x + F.relu(self.bn_x(self.W1(x) + aggr_messages))   # shape: (V, H)
        
        # --- Edge Update ---        
        w3_e = self.W3(e)  # shape: (E, H)
        w4_x_dest = self.W4(x[dest])  # shape: (E, H)
        w5_x_src = self.W5(x[src])  # shape: (E, H)
        
        e_new = e + F.relu(self.bn_e(w3_e + w4_x_dest + w5_x_src))   # shape: (E, H)

        return x_new, e_new