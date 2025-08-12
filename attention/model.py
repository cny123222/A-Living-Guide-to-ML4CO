import torch
from torch import Tensor
from ml4co_kit import BaseModel, TSPSolver
from attention.env import AttentionEnv
from attention.encoder import AttentionEncoder
from attention.decoder import AttentionDecoder
from attention.policy import AttentionPolicy
    
    
class AttentionModel(BaseModel):
    def __init__(
        self, 
        env: AttentionEnv,
        encoder: AttentionEncoder,
        decoder: AttentionDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        super(AttentionModel, self).__init__(
            env=env,
            # The main model to be trained
            model=AttentionPolicy(
                env=env,
                encoder=encoder,
                decoder=decoder,
            ),
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.to(self.env.device)
        
        # A separate baseline model for REINFORCE baseline
        self.baseline_model = AttentionPolicy(
            env=env,
            encoder=encoder,
            decoder=decoder,
        )
        self.baseline_model.eval()  # Set to evaluation mode permanently
        self.update_baseline()  # Initialize baseline with policy weights
        
    def update_baseline(self):
        """Copies the weights from the policy model to the baseline model."""
        self.baseline_model.load_state_dict(self.model.state_dict())
        
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training, validation, and testing.
        """
        self.env.mode = phase
        # unpack batch data
        points, ref_tours = batch
        # points: (batch_size, num_nodes, 2)
        # ref_tours: (batch_size, num_nodes + 1)
        if phase == "train":
            # --- 1. Policy Rollout (stochastic) ---
            # Gradients are tracked for this rollout.
            self.policy_model.train() # Ensure model is in training mode
            reward, sum_log_probs, _ = self.policy_model(points, mode='sampling')
            policy_cost = -reward  # Reward is negative tour length
            
            # --- 2. Baseline Rollout (greedy) ---
            # No gradients are needed for the baseline.
            with torch.no_grad():
                reward, _, _ = self.baseline_model(points, mode='greedy')
                baseline_cost = -reward
                
            # --- 3. Calculate REINFORCE Loss ---
            # The advantage is the gap between the sampled solution and the greedy baseline.
            advantage = policy_cost - baseline_cost
            # The loss is the mean of advantage-weighted negative log-probabilities.
            loss = (advantage * sum_log_probs).mean()
        elif phase == "val":
            with torch.no_grad():
                self.policy_model.eval() # Set model to evaluation mode
                
                # Evaluate the policy model
                _, _, tours = self.policy_model(points, mode='greedy')
                costs_avg, _, gap_avg, _ = self.evaluate(points, tours, ref_tours)
                
                _, _, baseline_tours = self.baseline_model(points, mode='greedy')
                baseline_costs_avg, _, _, _ = self.evaluate(points, baseline_tours, ref_tours)

        # --- 4. Logging ---
        metrics = {f"{phase}/loss": loss}
        # print(f"{phase} loss: {loss.item()}")
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg, "val/gap_avg": gap_avg, "val/baseline_costs_avg": baseline_costs_avg})
        for k, v in metrics.items():
            self.log(k, float(v), prog_bar=True, on_epoch=True, sync_dist=True)
        # return
        return loss if phase == "train" else metrics   
    
    def on_validation_epoch_end(self, outputs):
        # Aggregate the costs from all validation batches
        avg_policy_cost = torch.cat([x['val/costs_avg'] for x in outputs]).mean()
        avg_baseline_cost = torch.cat([x['val/baseline_costs_avg'] for x in outputs]).mean()
        # Baseline Update
        if avg_policy_cost < avg_baseline_cost:
            self.update_baseline()

    def evaluate(self, x: Tensor, tours: Tensor, ref_tours: Tensor):
        """
        Evaluate the model's performance on a given set of tours.
        
        Args:
            x: (batch_size, num_nodes, 2) tensor representing node coordinates.
            tours: (batch_size, num_nodes+1) tensor representing predicted tours.
            ref_tours: (batch_size, num_nodes+1) tensor representing reference tours.
        
        Returns:
            costs_avg: Average cost of the predicted tours.
            ref_costs_avg: Average cost of the reference tours.
            gap_avg: Average gap between predicted and reference tours.
            gap_std: Standard deviation of the gap.
        """
        x = x.cpu().numpy()
        tours = tours.cpu().numpy()
        ref_tour = ref_tour.cpu().numpy()
            
        solver = TSPSolver()
        solver.from_data(points=x, tours=tours, ref=False)
        solver.from_data(tours=ref_tours, ref=True)
        costs_avg, ref_costs_avg, gap_avg, gap_std = solver.evaluate(calculate_gap=True)
        return costs_avg, ref_costs_avg, gap_avg, gap_std