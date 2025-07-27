from torch import Tensor, nn

class Embedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Embedder, self).__init__()
        self.node_embed = nn.Linear(2, hidden_dim, bias=True)
        self.edge_embed = nn.Linear(1, hidden_dim, bias=True)
        
    def forward(self, x: Tensor, e: Tensor):
        """
        Args:
            x: (V, 2) nodes_feature (node coords)
            e: (E,) edges_feature (distance matrix)
        Return:
            x: (V, H)
            e: (E, H)
        """  
        x = self.node_embed(x) 
        e = self.edge_embed(e)
        return x, e