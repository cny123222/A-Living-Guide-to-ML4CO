import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiHeadMaskedCrossAttention(nn.Module):
    """
    Implements a Multi-Head Cross-Attention layer with masking.

    This layer is designed for a decoder that needs to attend to the output of an
    encoder. It takes a single context vector as the query source and a sequence
    of encoder outputs as the key and value source. It also supports masking to
    prevent attention to nodes that have already been visited in TSP.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadMaskedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Query, Key, Value, and the final output projection
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, context_query: Tensor, encoder_outputs: Tensor, mask: Tensor = None):
        """
        Forward pass for the Multi-Head Masked Cross-Attention layer.

        Args:
            context_query (torch.Tensor): The query tensor, typically derived from the decoder's state.
                                          Shape: (batch_size, 1, embed_dim).
            encoder_outputs (torch.Tensor): The key and value tensor, typically the output from the encoder.
                                            Shape: (batch_size, num_nodes, embed_dim).
            mask (torch.Tensor, optional): A boolean or 0/1 tensor to mask out certain keys.
                                           A value of 0 indicates the position should be masked.
                                           Shape: (batch_size, num_nodes).

        Returns:
            A tuple containing:
            - output (torch.Tensor): The attention-weighted output vector.
                                     Shape: (batch_size, 1, embed_dim).
            - attn_weights (torch.Tensor): The attention weights.
                                           Shape: (batch_size, num_heads, 1, num_nodes).
        """
        batch_size = context_query.shape[0]
        num_nodes = encoder_outputs.shape[1]

        # 1. Project Q from the context query and K, V from the encoder outputs.
        Q = self.fc_q(context_query)    # Shape: (batch_size, 1, embed_dim)
        K = self.fc_k(encoder_outputs)  # Shape: (batch_size, num_nodes, embed_dim)
        V = self.fc_v(encoder_outputs)  # Shape: (batch_size, num_nodes, embed_dim)

        # 2. Reshape and transpose for multi-head processing.
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)          # Shape: (batch_size, num_heads, 1, head_dim)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)

        # 3. Compute scaled dot-product attention scores.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, num_heads, 1, num_nodes)

        # 4. Apply the mask before the softmax step.
        if mask is not None:
            # Reshape mask for broadcasting: (batch_size, num_nodes) -> (batch_size, 1, 1, num_nodes)
            mask_reshaped = mask.unsqueeze(1).unsqueeze(2)
            # Fill masked positions (where mask is 0) with a very small number.
            attn_scores = attn_scores.masked_fill(mask_reshaped == 0, -1e9)
        
        # 5. Scale scores, apply softmax, and compute the context vector.
        scaled_attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scaled_attn_scores, dim=-1) # Shape: (batch_size, num_heads, 1, num_nodes)
        context = torch.matmul(attn_weights, V)              # Shape: (batch_size, num_heads, 1, head_dim)

        # 6. Concatenate heads and pass through the final linear layer.
        context = context.transpose(1, 2).contiguous()  # Shape: (batch_size, 1, num_heads, head_dim)
        context = context.view(batch_size, 1, self.embed_dim)  # Shape: (batch_size, 1, embed_dim)
        output = self.fc_out(context)  # Shape: (batch_size, 1, embed_dim)

        return output, attn_weights