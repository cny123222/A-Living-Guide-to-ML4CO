import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create separate linear layers for Query, Key, and Value
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        
        # Create the final fully connected output layer
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: Tensor):
        """
        Forward pass for the Multi-Head Self-Attention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_nodes, embed_dim).
        """
        batch_size = x.shape[0]
        
        # 1. Project input into Q, K, V using separate linear layers
        Q = self.fc_q(x)  # Shape: (batch_size, num_nodes, embed_dim)
        K = self.fc_k(x)  # Shape: (batch_size, num_nodes, embed_dim)
        V = self.fc_v(x)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # 2. Split the embed_dim into num_heads and head_dim
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        
        # 3. Calculate scaled dot-product attention
        # Calculate the dot product of Q and K
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, num_heads, num_nodes, num_nodes)
        # Scale the attention scores
        scaled_attn_scores = attn_scores / math.sqrt(self.head_dim)
        # Apply softmax to get the attention weights
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)  # Shape: (batch_size, num_heads, num_nodes, num_nodes)
        # Multiply the weights by V to get the context vector
        context = torch.matmul(attn_weights, V)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        
        # 4. Concatenate the attention heads' outputs
        # First, transpose to bring num_nodes and num_heads dimensions together
        context = context.transpose(1, 2).contiguous()  # Shape: (batch_size, num_nodes, num_heads, head_dim)
        # Then, reshape to combine the last two dimensions
        context = context.view(batch_size, -1, self.embed_dim)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # 5. Pass the concatenated context vector through the final linear layer
        output = self.fc_out(context)  # Shape: (batch_size, num_nodes, embed_dim)

        return output