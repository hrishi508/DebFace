from torch import nn

from backbones.encoder import Encoder
from backbones.decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(cfg)
        self.decoder = Decoder()
        self.cfg = cfg

    def forward(self, x):
        tmp = self.encoder(x)
        f_ID = tmp[:, self.cfg.embedding_size - 512 : self.cfg.embedding_size]
        y = self.decoder(f_ID)

        return y