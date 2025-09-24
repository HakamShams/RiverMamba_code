# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.autograd as autograd

# ------------------------------------------------------------------

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss(reduction='none') 
    def forward(self, input, target, weights=None):
        if weights is None:
            return self.loss(input, target).mean() # shape: (B, F, 1)
        else:
            return (self.loss(input, target) * weights[:, :, None]).mean()

class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, x_thr):
        return (x >= x_thr).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
        # return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x, x_thr):
        x = STEFunction.apply(x, x_thr)
        return x


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = nn.L1Loss(reduction='none')
        self.ste = StraightThroughEstimator()
    def forward(self, input, target, thresholds, weights=None):

        input = self.ste(input.repeat(1, 1, 9), thresholds)
        target = self.ste(target.repeat(1, 1, 9), thresholds)

        B, P, _ = target.shape

        N = B*P

        if weights is None:
            #for i in range(9):
            #    print(torch.histc(target[:, :, i], bins=2))
            weights = target.clone().requires_grad_(False)
            weights1 = torch.log(((N - torch.sum(target, dim=(0, 1))) / N) ** -0.5 + 1.)
            #weights1 = ((N - torch.sum(target, dim=(0, 1))) / N) ** -0.5
            #weights1 = torch.log(weights1 + 1.1)
            weights2 = torch.log(((torch.sum(target, dim=(0, 1))) / N) ** -0.5 + 1.)
            #weights2 = ((torch.sum(target, dim=(0, 1))) / N) ** -0.5
            #weights2 = torch.log(weights2 + 1.1)
            weights = weights + weights2[None, None, :]
            weights[target == 1] = (target * weights1[None, None, :])[target == 1]
            loss = (torch.sum((self.loss(input, target) * weights), dim=(0, 1)) / N).sum()
        else:
            loss = (torch.sum((self.loss(input, target) * weights[:, :, None]), dim=(0, 1)) / N).sum()

        return loss


if __name__ == '__main__':

    print()

