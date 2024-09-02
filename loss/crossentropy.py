import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        weights = torch.Tensor([1.25, 0.85, 0.81, 1.28]).float().cuda()
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x, label=None):
        loss = self.ce(x, label)
        return loss
