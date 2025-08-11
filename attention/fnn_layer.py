from torch import Tensor, nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super(FeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Create the first linear layer
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # Create the second linear layer
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x: Tensor):
        """
        Forward pass for the Feed Forward Neural Network layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_nodes, embed_dim).
        """
        # Apply the first linear layer followed by ReLU activation
        x = F.relu(self.fc1(x))  # Shape: (batch_size, num_nodes, hidden_dim)
        # Apply the second linear layer
        output = self.fc2(x)  # Shape: (batch_size, num_nodes, embed_dim)
        
        return output