from torch import nn, Tensor
from ml4co_kit import BaseModel, TSPSolver
from .env import GNNEnv
from .encoder import GCNEncoder
from .decoder import GNNDecoder

class GNNModel(BaseModel):
    def __init__(
        self,
        env: GNNEnv,
        encoder: GCNEncoder,
        decoder: GNNDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        super(GNNModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env = env
        self.model = encoder
        self.decoder = decoder
        self.to(self.env.device)
        
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training, validation, and testing.
        """
        self.env.mode = phase
        # unpack batch data
        x, e, edge_index, ref_tour, ground_truth = batch
        # x: (B, V, H), e: (B, E, H)
        # edge_index: (B, 2, E), ref_tour: (B, V)
        # ground_truth: (B, E)
        e_pred = self.model(x, e, edge_index)  # shape: (B, E)
        loss = nn.CrossEntropyLoss()(e_pred, ground_truth)
        if phase == "val":
            tours = self.decoder.decode(e_pred, x, edge_index)
            costs_avg, _, gap_avg, _ = self.evaluate(x, tours, ref_tour)
        # log
        metrics = {f"{phase}/loss": loss}
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg, "val/gap_avg": gap_avg})
        for k, v in metrics.items():
            formatted_v = f"{v:.8f}"
            self.log(k, float(formatted_v), prog_bar=True, on_epoch=True, sync_dist=True)
        # return
        return loss if phase == "train" else metrics   
            
    def evaluate(self, x: Tensor, tours: Tensor, ref_tour: Tensor):
        """
        Evaluate the model's performance on a given set of tours.
        
        Args:
            x: (B, V, H) tensor representing node features.
            tours: (B, V) tensor representing predicted tours.
            ref_tour: (B, V) tensor representing reference tours.
        
        Returns:
            costs_avg: Average cost of the predicted tours.
            ref_costs_avg: Average cost of the reference tours.
            gap_avg: Average gap between predicted and reference tours.
            gap_std: Standard deviation of the gap.
        """
        solver = TSPSolver()
        solver.from_data(points=x, tours=tours, ref=False)
        solver.from_data(tours=ref_tour, ref=True)
        costs_avg, ref_costs_avg, gap_avg, gap_std = solver.evaluate(calculate_gap=True)
        return costs_avg, ref_costs_avg, gap_avg, gap_std