import logging
from typing import Dict, Optional, Union, Tuple, List

import torch
import torch.nn as nn

from RiverMamba.lstm_repo.models_lstm.fc import FC

LOGGER = logging.getLogger(__name__)

_EMBEDDING_TYPES = ['full_model', 'hindcast', 'forecast']

class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs without relying on a config object.
    The input layer is doing the embedding 

    Parameters
    ----------
    dynamic_inputs : (batch, time step, num of features)
        Dynamic input features. It can be a list of input feature sizes or a dictionary where keys are frequencies.
    static_attributes : List[int]
        List of sizes for static attributes.
    statics_embedding_spec : Optional[Dict]
        Embedding specification for static inputs (optional).
        An example -- define embedding network for static inputs -- should be in this format:
        statics_embedding:
        type: fc
        # define number of neurons per layer in the FC network used as embedding network
        hiddens:
            - 30
            - 20
            - 64
        # activation function of embedding network
        activation: tanh
        # dropout applied to embedding network
        dropout: 0.0
    dynamics_embedding_spec : Optional[Dict]
        Embedding specification for dynamic inputs (optional). -- same as statics_embedding_spec
    embedding_type : str
        Embedding type, must be one of 'full_model', 'hindcast', 'forecast'.
    head : str
        Specifies the output type, used to adjust the output size.
    """

    """
    Output shape:

    """

    def __init__(
            self,
            dynamic_inputs1_size,  # already combined glofas and era5 and cpc
            static_attributes_size, 
            statics_embedding_spec,
            dynamics_embedding_spec,
            embedding_type: str = 'hindcast',
            head: str = 'default'):
        super(InputLayer, self).__init__()

        if embedding_type not in _EMBEDDING_TYPES:
            raise ValueError(
                f'Embedding type {embedding_type} is not recognized. '
                f'Must be one of: {_EMBEDDING_TYPES}.'
            )
        self.embedding_type = embedding_type

        # Determine dynamic input size -- number of features
        ## using torch --- (batch, time step, num of features)  --  considering the last dimension 
        dynamics_input_size = dynamic_inputs1_size 
        # Determine static input size -- this is num of input features
        statics_input_size = static_attributes_size

        # Create embedding networks
        self.statics_embedding, self.statics_output_size = \
            self._get_embedding_net(statics_embedding_spec, statics_input_size, 'statics') # return the embedding layer
        self.dynamics_embedding, self.dynamics_output_size = \
            self._get_embedding_net(dynamics_embedding_spec, dynamics_input_size, 'dynamics')

        # Output size including dynamics, statics, and autoregressive inputs
        self.output_size = self.dynamics_output_size + self.statics_output_size
        if head.lower() == "umal":
            self.output_size += 1

    @staticmethod
    def _get_embedding_net(embedding_spec: Optional[Dict], input_size: int, purpose: str) -> Tuple[nn.Module, int]:
        """Create embedding network based on the given specification."""
        if embedding_spec is None:
            return nn.Identity(), input_size

        if input_size == 0:
            raise ValueError(f'Cannot create {purpose} embedding layer with input size 0')

        emb_type = embedding_spec['type'].lower()
        if emb_type != 'fc':
            raise ValueError(f'{purpose} embedding type {emb_type} not supported.')

        hiddens = embedding_spec['hiddens']
        if len(hiddens) == 0:
            raise ValueError(f'{purpose} embedding "hiddens" must be a list of hidden sizes with at least one entry')

        dropout = embedding_spec['dropout']
        activation = embedding_spec['activation']

        emb_net = FC(input_size=input_size, hidden_sizes=hiddens, activation=activation, dropout=dropout) # input size means num of features of the input

        return emb_net, emb_net.output_size

    # need changes here -- dynamic and static features should use the one from __init__s
    def forward(self,x_dyanmic, x_static, concatenate_input: bool = True, concatenate_output: bool = True):
        """Perform a forward pass on the input layer.

        Parameters
        ----------
        data : Dict[str, torch.Tensor] -- TO DO
            The input data.
            The acutal data format we need for input layer -- x_glofas, x_era5, x_static -- first two are dynamic inputs, the last is static
        concatenate_output : bool, optional
            If True (default), the forward method will concatenate the static inputs to each dynamic time step. -- should be the normal case
            If False, the forward method will return a tuple of (dynamic, static) inputs.

        Returns
        -------
        torch.Tensor, shape --> (batch, time, num of hidden layers from EM*2 (dynamic+static))
            If `concatenate_output` is True, a single tensor is returned. Else, a tuple with one tensor of dynamic
            inputs and one tensor of static inputs.
        """
        # glofas/era5 -- (batch, time steps, feature)
 
        x_d = x_dyanmic # for input: both glofas+era5+cpc; for output: hres or era5
            
        x_s = x_static # shape: (batch size, num of static features)

        # Process dynamic inputs through the dynamics embedding network
        dynamics_out = self.dynamics_embedding(x_d)
        
        statics_out = None
        if x_s is not None:
            statics_out = self.statics_embedding(x_s)

        if not concatenate_output:
            ret_val = dynamics_out, statics_out
        else:
            if statics_out is not None:
                
                statics_out = statics_out.unsqueeze(1).repeat(1, dynamics_out.shape[1], 1) # update the shape of embedded static variables to match the dynamic embedding
                ret_val = torch.cat([dynamics_out, statics_out], dim=-1)
            else:
                ret_val = dynamics_out
        

        return ret_val

    def __getitem__(self, item: str) -> nn.Module:
        if item == "statics_embedding":
            return self.statics_embedding
        elif item == "dynamics_embedding":
            return self.dynamics_embedding
        else:
            raise KeyError(f"Cannot access {item} on InputLayer")

