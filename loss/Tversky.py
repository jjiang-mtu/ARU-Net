"""
Tversky loss
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # Definition of dice coefficient
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1)+
                                            0.3 * (pred * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # Return dice distance
        return torch.clamp((1 - dice).mean(), 0, 2)
