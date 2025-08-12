from .env import AttentionEnv
from .encoder import AttentionEncoder
from .decoder import AttentionDecoder
from .model import AttentionModel
from ml4co_kit import Trainer


if __name__ == "__main__":
    model = AttentionModel(
        env=AttentionEnv(
            mode="train",
            train_batch_size=32,
            val_batch_size=4,
            train_path="data/tsp20/tsp20_gaussian_train.txt",
            val_path="data/tsp20/tsp20_gaussian_val.txt",
            device="cuda",
        ),
        encoder=AttentionEncoder(
            embed_dim=128,
            num_heads=8,
            hidden_dim=512,
            num_layers=3,
        ),
        decoder=AttentionDecoder(
            embed_dim=128,
            num_heads=8,
        ),
    )
    
    trainer = Trainer(model=model, devices=[0], max_epochs=20)
    trainer.model_train()