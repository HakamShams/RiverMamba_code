import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class PE_1D(nn.Module):
    """ Positional Encoding """

    def __init__(self, d_model: int = 32, n_days: int = 366, device: str = 'cuda'):
        super(PE_1D, self).__init__()

        """
        Parameters
        ----------
        d_model : int (default 256)
            number of dimensions for encoding
        n_days : int (default 366)
            number of days
        device : str (default cuda)
            device GPU or CPU
        """

        self.d_model = d_model
        self.n_days = n_days
        self.device = device

        pe = torch.zeros(n_days, d_model).to(device)

        # precompute the encoding
        for pos in range(n_days):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10 ** ((2 * i) / d_model)))

        # store in buffer for fast access
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input day of the year x_t [N]
        """
        x = Variable(self.pe[x - 1, :]).to(self.device)

        return x


if __name__ == '__main__':

    m = PE_1D(d_model=128, n_days=366, device='cuda')

    x = torch.arange(1, 366 + 1).cuda()

    x = x[None, None, :].repeat(2, 4, 1)

    y = m(x)
    print(y.shape)
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    plt.imshow(y.cpu().detach().numpy()[0, 1, :, :].T)
    plt.show()



