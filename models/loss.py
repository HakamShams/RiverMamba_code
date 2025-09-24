# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
import torch
import torch.nn as nn

# ------------------------------------------------------------------

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, input, target, weights=None):
        if weights is None:
            loss = self.loss(input, target) * weights
            loss = torch.nan_to_num(loss)
            return loss.mean()
        else:
            loss = self.loss(input, target) * weights
            loss = torch.nan_to_num(loss)
            return loss.mean()


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, input, target, weights=None):
        if weights is None:
            loss = self.loss(input, target) * weights
            loss = torch.nan_to_num(loss)
            return loss.mean()
        else:
            loss = self.loss(input, target) * weights
            loss = torch.nan_to_num(loss)
            return loss.mean()

