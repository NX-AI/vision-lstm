import torch.nn as nn
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, reduction="mean"):
        return F.smooth_l1_loss(pred, target, reduction=reduction, beta=self.beta)
