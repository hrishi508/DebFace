import torch
from torch import nn

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Am_softmax(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332):
        super(Am_softmax,  self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        
        # initial kernel
        self.kernel.data.uniform_(-1,  1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35 # additive margin recommended by the paper
        self.s = 30. # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1) # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1) #size=(B, 1)
        index = cos_theta.data * 0.0 #size=(B, Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index.bool()] = phi[index.bool()] #only change the correct predicted output
        output *= self.s # scale up in order to make softmax work,  first introduced in normface
        return output