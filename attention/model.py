import torch
from ml4co_kit import BaseModel
from attention.env import AttentionEnv
from attention.encoder import AttentionEncoder
from attention.decoder import AttentionDecoder
    
    
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
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env = env
        self.model = encoder
        self.decoder = decoder
        self.to(self.env.device)
        
    