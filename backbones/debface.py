from torch import nn
import torch

from backbones.encoder import Encoder
from backbones.classifier import Classifier

class DebFace(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(cfg)
        self.C_gender = Classifier(num_classes=cfg.n_gender_classes)
        self.C_age = Classifier(num_classes=cfg.n_age_classes)
        self.C_race = Classifier(num_classes=cfg.n_race_classes)
        self.C_id = Classifier(num_classes=cfg.n_id_classes)
        self.C_distr = Classifier(cfg.embedding_size)

    def forward(self, x):
        x = self.encoder(x)

        f_G = x[:, : 512] * 1.0
        f_A = x[:, 512 : 1024] * 1.0
        f_R = x[:, 1024 : 1536] * 1.0
        f_ID = x[:, 1536 : 2048] * 1.0
        f_Joint = x * 1.0

        r = torch.randperm(4)
        f_Shuffled = x * 1.0

        tmp1 = f_Shuffled[:, : 512] * 1.0
        tmp2 = f_Shuffled[:, 512 : 1024] * 1.0
        tmp3 = f_Shuffled[:, 1024 : 1536] * 1.0
        tmp4 = f_Shuffled[:, 1536 : 2048] * 1.0

        f_Shuffled[:, (512 * r[0].item()) : (512 * (r[0].item() + 1))] = tmp1
        f_Shuffled[:, (512 * r[1].item()) : (512 * (r[1].item() + 1))] = tmp2
        f_Shuffled[:, (512 * r[2].item()) : (512 * (r[2].item() + 1))] = tmp3
        f_Shuffled[:, (512 * r[3].item()) : (512 * (r[3].item() + 1))] = tmp4

        out_G = self.C_gender(f_G)
        out_A = self.C_age(f_A)
        out_R = self.C_race(f_R)
        out_ID = self.C_id(f_ID)
        out_Distr1 = self.C_distr(f_Joint)
        out_Distr2 = self.C_distr(f_Shuffled)

        return out_G, out_A, out_R, out_ID, out_Distr1, out_Distr2
