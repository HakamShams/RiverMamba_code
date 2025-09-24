import logging
from typing import Dict

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

def get_head(head_type: str, n_in: int, n_out: int, output_activation: str = 'linear') -> nn.Module:
    """Get specific head module, depending on the provided head type.

    Parameters
    ----------
    head_type : str
        Type of the model head (e.g., "regression", "gmm", "umal", "cmal").
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.
    output_activation : str, optional
        The activation function to be used in the output layer for regression heads -- set to linear by default

    Returns
    -------
    nn.Module
        The model head, as specified by the provided type.
    """
    if head_type.lower() == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=output_activation)
    elif head_type.lower() == "gmm":
        head = GMM(n_in=n_in, n_out=n_out)
    elif head_type.lower() == "umal":
        head = UMAL(n_in=n_in, n_out=n_out)
    elif head_type.lower() == "cmal":
        head = CMAL(n_in=n_in, n_out=n_out)
    else:
        raise NotImplementedError(f"{head_type} not implemented or not linked in `get_head()`")

    return head


class Regression(nn.Module):
    """Single-layer regression head with different output activations.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation : str, optional
        Output activation function. Supported values are {'linear', 'relu', 'softplus'}. Defaults to 'linear'.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Perform a forward pass on the Regression head.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        
        """
        #print (self.net(x).shape)
        return self.net(x)


class GMM(nn.Module):
    """Gaussian Mixture Density Network

    A mixture density network with Gaussian distribution as components. Good references are [#]_ and [#]_. 

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 3 times the number of components.
    n_hidden : int, optional
        Size of the hidden layer. Defaults to 100.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(GMM, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a GMM head forward pass.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the mixture parameters ('mu', 'sigma', 'pi').
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        # split output into mu, sigma, and weights
        mu, sigma, pi = h.chunk(3, dim=-1)

        return {'mu': mu, 'sigma': torch.exp(sigma) + self._eps, 'pi': torch.softmax(pi, dim=-1)}


class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int, optional
        Size of the hidden layer. Defaults to 100.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a CMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing mixture parameters ('mu', 'b', 'tau', 'pi').
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        m = m_latent
        b = self._softplus(b_latent) + self._eps
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}


class UMAL(nn.Module):
    """Uncountable Mixture of Asymmetric Laplacians.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 2 times the output size.
    n_hidden : int, optional
        Size of the hidden layer. Defaults to 100.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(UMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._upper_bound_scale = 0.5
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a UMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the means ('mu') and scale parameters ('b').
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent = h.chunk(2, dim=-1)

        m = m_latent
        b = self._upper_bound_scale * torch.sigmoid(b_latent) + self._eps
        return {'mu': m, 'b': b}

