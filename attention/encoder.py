from torch import Tensor, nn
from attention.attn_layer import MultiHeadSelfAttention
from attention.ff_layer import FeedForward


class AttentionLayer(nn.Module):
    """
    A single Attention Layer that follows the structure from the image.
    It consists of a Multi-Head Attention sublayer and a Feed-Forward sublayer.
    Each sublayer is followed by a skip connection and Batch Normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        
        # Sublayer 1: Multi-Head Attention
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # Sublayer 2: Feed-Forward Network
        self.ff = FeedForward(embed_dim, hidden_dim)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, x: Tensor):
        """
        Forward pass for one complete attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # --- Multi-Head Attention Sublayer ---
        # 1. Apply MHA
        mha_output = self.mha(x)
        
        # 2. Add skip connection
        sublayer1_input = x + mha_output
        
        # 3. Apply Batch Normalization
        # Permute from (batch, seq_len, features) to (batch, features, seq_len) for BN
        sublayer1_output = self.bn1(sublayer1_input.permute(0, 2, 1)).permute(0, 2, 1)

        # --- Feed-Forward Sublayer ---
        # 1. Apply Feed-Forward network
        ff_output = self.ff(sublayer1_output)
        
        # 2. Add skip connection
        sublayer2_input = sublayer1_output + ff_output
        
        # 3. Apply Batch Normalization
        # Permute for BN and then permute back
        output = self.bn2(sublayer2_input.permute(0, 2, 1)).permute(0, 2, 1)

        return output  # Shape: (batch_size, num_nodes, embed_dim)


class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int):
        super(AttentionEncoder, self).__init__()
        # Embedding layer
        self.embed = nn.Linear(2, embed_dim)
        
        # Stack of identical Attention Layers
        self.layers = nn.ModuleList([
            AttentionLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x: Tensor):
        """
        Forward pass for the Encoder.
        Args:
            x (torch.Tensor): Coordinates of nodes with shape (batch_size, num_nodes, 2).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_nodes, embed_dim).
        """
        # Embed the input coordinates
        x = self.embed(x)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # Pass through multiple attention layers
        for layer in self.layers:
            x = layer(x)  # Shape: (batch_size, num_nodes, embed_dim)
            
        return x  # Shape: (batch_size, num_nodes, embed_dim)