from torch import Tensor, nn

class OutLayer(nn.Module):
    def __init__(self, hidden_dim: int, layer_num: int):
        """
        Args:
            hidden_dim: The dimension of the input edge features.
            layer_num: The number of layers in the MLP.
        """
        super(OutLayer, self).__init__()
        mlp_layers = []
        if layer_num == 1:
            mlp_layers.append(nn.Linear(hidden_dim, 1))
        else:
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            for _ in range(layer_num - 2):
                mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, e_final: Tensor):
        """     
        Args:
            e_final: (E, H) Final edge features
        Returns:
            prob: (E,) probablity of each edge being connected to the TSP tour
        """
        prob = self.mlp(e_final)  # shape: (E, 1)
        prob = prob.squeeze(-1)  # shape: (E, )
        return prob