import torch
from torch import nn


class PseudoHuberLoss(nn.Module):
    """The Pseudo-Huber loss."""

    reductions = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}
                
    def __init__(self, c=0.01, reduction='mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def extra_repr(self):
        return f'beta={self.beta:g}, reduction={self.reduction!r}'

    def forward(self, input, target):
        output = torch.sqrt((input - target)**2. + self.c**2.) - self.c
        return self.reductions[self.reduction](output)


def huber_loss(x, y, c = 0.01, reduction='mean'):
    reductions = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}
    c = torch.tensor(c, device=x.device)
    return reductions[reduction](torch.sqrt((x - y)**2. + c**2.) - c) # original
    # return reductions[reduction](torch.sqrt(torch.sqrt((x - y)**2. + 1e-5) + c**2.) - c) # modified
