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
        self.C_distr = Classifier(cfg.embedding_size, num_classes=cfg.n_distr_classes)

    def forward(self, x):
        x = self.encoder(x)

        f_G = x[:, : 512]
        f_A = x[:, 512 : 1024]
        f_R = x[:, 1024 : 1536]
        f_ID = x[:, 1536 : 2048]

        r = torch.randperm(4)
        f_Shuffled = torch.zeros_like(x)

        f_Shuffled[:, (512 * r[0].item()) : (512 * (r[0].item() + 1))] = f_G
        f_Shuffled[:, (512 * r[1].item()) : (512 * (r[1].item() + 1))] = f_A
        f_Shuffled[:, (512 * r[2].item()) : (512 * (r[2].item() + 1))] = f_R
        f_Shuffled[:, (512 * r[3].item()) : (512 * (r[3].item() + 1))] = f_ID

        out_G1  = self.C_gender(f_G)
        out_A1  = self.C_gender(f_A)
        out_R1  = self.C_gender(f_R)
        out_ID1 = self.C_gender(f_ID)

        out_G2  = self.C_age(f_G)
        out_A2  = self.C_age(f_A)
        out_R2  = self.C_age(f_R)
        out_ID2 = self.C_age(f_ID)

        out_G3  = self.C_race(f_G)
        out_A3  = self.C_race(f_A)
        out_R3  = self.C_race(f_R)
        out_ID3 = self.C_race(f_ID)

        out_G4  = self.C_id(f_G)
        out_A4  = self.C_id(f_A)
        out_R4  = self.C_id(f_R)
        out_ID4 = self.C_id(f_ID)

        out_Distr1 = self.C_distr(x)
        out_Distr2 = self.C_distr(f_Shuffled)

        return out_G1, out_G2, out_G3, out_G4, out_A1, out_A2, out_A3, out_A4, out_R1, out_R2, out_R3, out_R4, out_ID1, out_ID2, out_ID3, out_ID4, out_Distr1, out_Distr2

class DebFaceWithoutRace(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(cfg)
        self.C_gender = Classifier(num_classes=cfg.n_gender_classes)
        self.C_age = Classifier(num_classes=cfg.n_age_classes)
        self.C_id = Classifier(num_classes=cfg.n_id_classes)
        self.C_distr = Classifier(cfg.embedding_size, num_classes=cfg.n_distr_classes)

    def forward(self, x):
        x = self.encoder(x)

        f_G = x[:, : 512]
        f_A = x[:, 512 : 1024]
        f_ID = x[:, 1024 : 1536]

        r = torch.randperm(3)
        f_Shuffled = torch.zeros_like(x)

        f_Shuffled[:, (512 * r[0].item()) : (512 * (r[0].item() + 1))] = f_G
        f_Shuffled[:, (512 * r[1].item()) : (512 * (r[1].item() + 1))] = f_A
        f_Shuffled[:, (512 * r[2].item()) : (512 * (r[2].item() + 1))] = f_ID

        out_G1  = self.C_gender(f_G)
        out_A1  = self.C_gender(f_A)
        out_ID1 = self.C_gender(f_ID)

        out_G2  = self.C_age(f_G)
        out_A2  = self.C_age(f_A)
        out_ID2 = self.C_age(f_ID)

        out_G3  = self.C_id(f_G)
        out_A3  = self.C_id(f_A)
        out_ID3 = self.C_id(f_ID)

        out_Distr1 = self.C_distr(x)
        out_Distr2 = self.C_distr(f_Shuffled)

        return out_G1, out_G2, out_G3, out_A1, out_A2, out_A3, out_ID1, out_ID2, out_ID3, out_Distr1, out_Distr2
