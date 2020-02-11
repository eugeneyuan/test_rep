import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.miscs import make_one_hot


def _flatten(tensor):
    r"""Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    permuted = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return permuted.contiguous().view(C, -1)


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, num_class=2, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_class = num_class
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        if self.ignore_index is not None:
            mask = targets.clone().ne_(self.ignore_index).requires_grad_(False).to(inputs.device)
            inputs = inputs * mask.unsqueeze(1).float()
            targets = targets * mask
        i_flat = _flatten(inputs)[1:].float()  # ignore 0 class
        targets = make_one_hot(targets, self.num_class)
        t_flat = _flatten(targets)[1:].float()

        intersection = (i_flat * t_flat).sum(dim=1)
        return torch.mean(1 - intersection.mul(2.0).add(self.smooth).div(
            i_flat.sum(dim=1) + t_flat.sum(dim=1) + self.smooth))


class FocalLoss(nn.Module):

    def __init__(self, num_class=2, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if isinstance(self.alpha, (list, tuple, np.ndarray)):
            self.alpha = torch.tensor(self.alpha)
        else:
            self.alpha = torch.ones(self.num_class, 1)

    def forward(self, inputs, targets):
        pt = F.softmax(inputs, dim=1)
        if self.ignore_index is not None:
            mask = targets.clone().ne_(self.ignore_index).requires_grad_(False).to(inputs.device)
            pt = pt * mask.unsqueeze(1).float()
            targets = targets * mask
        pt_flat = _flatten(pt).float()
        t_flat = _flatten(make_one_hot(targets, self.num_class)).float()
        alpha = self.alpha.to(t_flat.device).float()
        loss = - alpha * torch.pow((1 - pt_flat), self.gamma) * torch.log(pt_flat + 1e-10) * t_flat
        if self.reduction == 'mean':
            return torch.mean(loss.sum(1) / (t_flat.sum(1) + 1e-10))
        else:
            return torch.sum(loss.sum(1) / (t_flat.sum(1) + 1e-10))


class TotalVariationLoss(nn.Module):

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        batch_size, _, d, h, w = x.size()
        d_tv = torch.pow(x[:, :, 1:, :, :] - x[:, :, :d - 1, :, :], 2).sum() / ((d - 1) * h * w)
        h_tv = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :h - 1, :], 2).sum() / (d * (h - 1) * w)
        w_tv = torch.pow(x[:, :, :, 1:, :] - x[:, :, :, :w - 1, :], 2).sum() / (d * h * (w - 1))
        return (d_tv + h_tv + w_tv) / batch_size

