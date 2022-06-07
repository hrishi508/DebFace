from torch import nn
from backbones import get_model

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.arcFace50 = get_model(
            cfg.network, dropout=0.0, fp16=False, num_features=cfg.embedding_size).to(cfg.device)
        self.ReLU = nn.ReLU(inplace=True) # Check the inplace arg

    def forward(self, x):
        x = self.arcFace50(x)
        x = self.ReLU(x) # Verify if a relu is needed here 
        return x


