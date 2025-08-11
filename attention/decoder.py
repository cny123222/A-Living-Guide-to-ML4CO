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
    
    
class Decoder(nn.Module):
    """
    Implements the Decoder for the Attention Model.

    At each step, it creates a context embedding based on the graph, the first
    node, and the previously visited node. It then uses two attention mechanisms:
    1. A multi-head "glimpse" to refine the context.
    2. A single-head mechanism with clipping to calculate the final output probabilities.
    """
    def __init__(self, embed_dim: int, num_heads: int, clip_value: float = 10.0):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.clip_value = clip_value

        # Learned placeholders for the first and last nodes at the initial step (t=1)
        self.v_first_placeholder = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.v_last_placeholder = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Projection layer for the concatenated context vector
        self.context_projection = nn.Linear(3 * embed_dim, embed_dim, bias=False)

        # The first attention mechanism: a multi-head "glimpse".
        self.glimpse_attention = MultiHeadMaskedCrossAttention(embed_dim, num_heads)
        
        # Layers for the final single-head attention mechanism to compute probabilities.
        self.final_q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.final_k_projection = nn.Linear(embed_dim, embed_dim, bias=False)


    def forward(self, encoder_outputs: Tensor, partial_tour: Tensor, mask: Tensor):
        """
        Performs a single decoding step.

        Args:
            encoder_outputs (torch.Tensor): The final node embeddings from the encoder.
                                            Shape: (batch_size, num_nodes, embed_dim).
            partial_tour (torch.Tensor): A tensor of node indices for the current partial tours.
                                         Shape: (batch_size, current_tour_length).
            mask (torch.Tensor): A tensor indicating which nodes are available to be visited.
                                 Shape: (batch_size, num_nodes).

        Returns:
            log_probs (torch.Tensor): The log-probabilities for selecting each node as the next step.
                                        Shape: (batch_size, num_nodes).
        """
        batch_size = encoder_outputs.shape[0]

        # 1. Construct the Context Embedding for the entire batch
        graph_embedding = encoder_outputs.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1, embed_dim)

        if partial_tour.size(1) == 0: # If this is the first step (t=1) for all instances
            # Use learned placeholders
            first_node_emb = self.v_first_placeholder.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
            last_node_emb = self.v_last_placeholder.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        else:
            # Get indices of the first and last nodes for each instance in the batch
            first_node_indices = partial_tour[:, 0]  # Shape: (batch_size,)
            last_node_indices = partial_tour[:, -1]  # Shape: (batch_size,)
            
            # Use torch.gather to select the corresponding embeddings
            first_node_emb = torch.gather(encoder_outputs, 1, first_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim))  # Shape: (batch_size, 1, embed_dim)
            last_node_emb = torch.gather(encoder_outputs, 1, last_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim))  # Shape: (batch_size, 1, embed_dim)

        # Concatenate the three components to form the raw context
        raw_context = torch.cat([graph_embedding, first_node_emb, last_node_emb], dim=2)  # Shape: (batch_size, 1, 3 * embed_dim)
        
        # Project the context to create the initial query
        context_query = self.context_projection(raw_context)  # Shape: (batch_size, 1, embed_dim)

        # 2. Perform the Multi-Head "Glimpse"
        glimpse_output, _ = self.glimpse_attention(
            context_query=context_query,
            encoder_outputs=encoder_outputs,
            mask=mask
        )  # Shape: (batch_size, 1, embed_dim)

        # 3. Calculate Final Log-Probabilities
        final_q = self.final_q_projection(glimpse_output)  # Shape: (batch_size, 1, embed_dim)
        final_k = self.final_k_projection(encoder_outputs)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # Calculate compatibility scores (logits)
        logits = torch.matmul(final_q, final_k.transpose(-2, -1)).squeeze(1) / math.sqrt(self.embed_dim)  # Shape: (batch_size, num_nodes)

        # Clip the logits before masking
        logits = self.clip_value * torch.tanh(logits)  # Shape: (batch_size, num_nodes)

        # Apply the mask again to ensure forbidden nodes are not chosen
        logits[mask == 0] = -torch.inf
        
        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, num_nodes)

        return log_probs