from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from .env import AttentionEnv
from .encoder import AttentionEncoder
from .decoder import AttentionDecoder


@dataclass
class StepState:
    """
    A data class to hold the state of the environment at each decoding step.
    This makes passing state information to the model cleaner.
    """
    current_node: Tensor = None  # Shape: (batch,)
    tours: Tensor = None  # Shape: (batch, time_step)
    mask: Tensor = None  # Shape: (batch, num_nodes)
    

class AttentionPolicy(nn.Module):
    def __init__(
        self,
        env: AttentionEnv,
        encoder: AttentionEncoder,
        decoder: AttentionDecoder,
    ):
        super(AttentionPolicy, self).__init__()
        self.env = env
        self.encoder = encoder
        self.decoder = decoder
        self.to(self.env.device)
        
    def forward(self, points: Tensor, mode: str = "sampling"):
        """
        Performs a full rollout to generate a tour for a batch of TSP instances.

        Args:
            points (torch.Tensor): Node coordinates for the batch.
                                    Shape: (batch_size, num_nodes, 2).
            mode (str): 'sampling' for stochastic rollout or 'greedy' for deterministic.

        Returns:
            A tuple containing:
            - reward (torch.Tensor): Reward for each instance in the batch. Shape: (batch_size,).
            - sum_log_probs (torch.Tensor): Sum of action log probabilities. Shape: (batch_size,).
            - tour (torch.Tensor): The decoded tour for each instance. Shape: (batch_size, num_nodes + 1).
        """
        batch_size = points.size(0)
        
        # Pre-computation step
        encoder_outputs = self.encoder(points)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # Initialize environment for this rollout
        state, reward, done = self.env.reset(points)
        
        # Perform the rollout
        sum_log_probs = torch.zeros(batch_size, device=self.env.device)
        while not done:
            log_probs = self.decoder(encoder_outputs, state.tours, state.mask)  # Shape: (batch_size, num_nodes)
            dist = Categorical(logits=log_probs)  # Create a categorical distribution from log probabilities
            if mode == "sampling":
                # Sample from the distribution
                selected_node = dist.sample()  # Shape: (batch_size,)
            elif mode == "greedy":
                selected_node = log_probs.argmax(dim=1)
            else:
                raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            
            sum_log_probs += dist.log_prob(selected_node)
            state, reward, done = self.env.step(selected_node)
            
        tour = state.tours  # Shape: (batch_size, num_nodes)
        start_node = tour[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)
        tour = torch.cat([tour, start_node], dim=1)  # Append the start node to the end of the tour
            
        return reward, sum_log_probs, tour