import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha # weight for the positive class
        self.gamma = gamma # focal-parameter, reduces the loss for already well-classified examples
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') # reduce=none to weight them manually afterwards
        pt = torch.exp(-bce_loss) # probability of the true class

        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha) # alpha for the positive class, 1-alpha for the negative class

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss # weighting of 'simpler' samples

        # choose how to reduce the loss across the batch
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
