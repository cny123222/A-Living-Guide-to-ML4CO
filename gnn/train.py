from .env import GNNEnv
from .encoder import GCNEncoder
from .decoder import GNNDecoder
from .model import GNNModel
from ml4co_kit import Trainer


if __name__ == "__main__":
    model = GNNModel(
        env=GNNEnv(
            mode="train",
            train_batch_size=32,
            val_batch_size=4,
            train_path="gnn/data/tsp20_gaussian_train.txt",
            val_path="gnn/data/tsp20_gaussian_val.txt",
            device="cuda",
        ),
        encoder=GCNEncoder(
            hidden_dim=64,
            gcn_layer_num=10,
            out_layer_num=3,
        ),
        decoder=GNNDecoder(
            decoding_type="greedy",
        ),
        learning_rate=1e-4,
    )
    
    trainer = Trainer(model=model, devices=[0], max_epochs=20)
    trainer.model_train()