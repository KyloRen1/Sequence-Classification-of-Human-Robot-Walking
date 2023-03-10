import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, label_smoothing=0.0, gamma=2, reduction='mean'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction, weight=self.weight, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss