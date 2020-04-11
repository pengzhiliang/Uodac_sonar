import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_uncertain_loss


@weighted_uncertain_loss
def uncertainty_kl_loss(pred, target, uncertainty, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * torch.exp(-uncertainty) * diff * diff / beta + 0.5 * uncertainty,
                       torch.exp(-uncertainty) * (diff - 0.5 * beta) + 0.5 * uncertainty)
    return loss


@LOSSES.register_module
class UncertainKLLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(UncertainKLLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                uncertainty,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * uncertainty_kl_loss(
            pred,
            target,
            uncertainty,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
