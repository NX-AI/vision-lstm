import torch
import torch.nn as nn

from ksuit.factory import MasterFactory


class ElementwiseLoss(nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = MasterFactory.get("loss").create(loss_function)

    def forward(self, pred, target, mask=None, reduction="mean"):
        assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"
        # unreduced loss
        loss = self.loss_function(pred, target, reduction="none")
        # apply mask
        if mask is not None:
            assert mask.dtype == torch.bool and loss.shape == mask.shape
            loss = loss[mask]
        # apply reduction
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        else:
            raise NotImplementedError
        return loss
