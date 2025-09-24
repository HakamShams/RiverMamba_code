import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Add the parent directory of models_lstm to sys.path
from RiverMamba.lstm_repo.models_lstm.inputlayer_lstm import InputLayer
from RiverMamba.lstm_repo.models_lstm.head_lstm import get_head
from RiverMamba.lstm_repo.models_lstm.fc import FC

class HandoffForecastLSTM(nn.Module):
    """
    A hindcast-forecast model that rolls out over the forecast horizon in the encoder-decoder LSTM fashion.

    This forecasting model uses a state-handoff mechanism to transition from a hindcast LSTM (bidirectional)
    to a forecast LSTM (unidirectional). The hindcast LSTM runs from past to the issue time, and its hidden and
    cell states are passed into a learnable handoff network. The transformed states are then used to initialize
    the forecast LSTM that generates predictions into the future.

    The dynamic input for the forecast stage excludes GloFAS.

    Parameters
    ----------
    glofas_inputs_size_h : int
        Number of hindcast GloFAS input features.
    era5_inputs_size_h : int
        Number of hindcast ERA5 input features.
    inputs_size_f : int
        Number of forecast input features (e.g., from HRES).
    cpc_inputs_size : int
        Number of hindcast CPC input features.
    static_attributes_size : int
        Number of static attributes per sample.
    hindcast_hidden_size : int
        Number of hidden units in the hindcast LSTM.
    forecast_hidden_size : int
        Number of hidden units in the forecast LSTM.
    output_dropout : float
        Dropout rate for the output head.
    output_size : int
        Number of target variables to predict.
    initial_forget_bias : int
        Initial bias value for the LSTM forget gate.
    static_embedding_spec : Dict
        Embedding specification for static inputs. Required to instantiate the InputLayer module.
        Example:
            statics_embedding:
                type: fc
                hiddens: [30, 20, 64]
                activation: tanh
                dropout: 0.0
    dynamic_embedding_spec : Dict
        Embedding specification for dynamic inputs. Same structure as static_embedding_spec.
    state_handoff_network_para : Dict
        Dictionary for the fully connected network that transforms LSTM hidden/cell states across the handoff.
        Should include keys:
            hiddens: List[int] (e.g., [64, 32])
            activation: str (e.g., 'tanh')
            dropout: float (e.g., 0.1)
    head_type : str
        Type of output head used for the forecast (e.g., 'regression', 'gmm', 'cmal').

    Notes
    -----
    This model is especially useful for hydrometeorological forecasting problems where a long hindcast history
    (e.g., GloFAS/ERA5/CPC) is available, but forecast inputs (e.g., HRES) differ in structure or source.
    
    Modified from Source: https://github.com/neuralhydrology/neuralhydrology
    """

    # specify submodules of the model that can later be used for finetuning. Names must match class attributes

    def __init__(self, 
                 glofas_inputs_size_h, # num of features
                 era5_inputs_size_h,
                 inputs_size_f, # could be era5 or hres
                 cpc_inputs_size,
                 static_attributes_size,
                 hindcast_hidden_size: int, # define the hindcast hidden network neurons
                 forecast_hidden_size: int, # define the forecast hidden network neurons
                 output_dropout: float, 
                 output_size: int, # target variables number
                 initial_forget_bias: int, # forget layer bias init
                 static_embedding_spec: Dict,
                 dynamic_embedding_spec: Dict,
                 state_handoff_network_para: Dict, 
                 head_type: str):
        
        super(HandoffForecastLSTM, self).__init__() #target_variables, head_type, n_distributions)

        self.static_embedding_spec = static_embedding_spec
        self.dynamic_embedding_spec = dynamic_embedding_spec
        self.initial_forget_bias = initial_forget_bias
        self.hindcast_hidden_size = hindcast_hidden_size
        self.forecast_hidden_size = forecast_hidden_size

        # Define embedding layers for both hindcast and forecast
        # In the input layer, the dynamic inputs are concatenated and passed to the embedding network
        # the Input layer only receive one dynamic size and one static size
        
        self.hindcast_embedding_net = InputLayer(
            glofas_inputs_size_h+era5_inputs_size_h+cpc_inputs_size,
            static_attributes_size,
            static_embedding_spec,
            dynamic_embedding_spec,
            embedding_type='hindcast')
        
        # In the input layer, the dynamic inputs are concatenated and passed to the embedding network
        # the Input layer only receive one dynamic size and one static size

        self.forecast_embedding_net = InputLayer(
            inputs_size_f, 
            static_attributes_size,
            static_embedding_spec,
            dynamic_embedding_spec,
            embedding_type='forecast'
        )

        # Define hindcast LSTM 
        
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size, 
            ### input size is the shape of input from the dynamic embedding -- (seq_length(time steps), batch, num of features) or (batch, seq_length(time steps),num of features)
            hidden_size=hindcast_hidden_size, # define the hidden size of the LSTM layer (num of neurons of LSTM hidden layers) -- int, a value,
            bidirectional=True
        )

        # define the Forecast LSTM:
        self.forecast_lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size, 
            ### input size is the shape of input from the dynamic embedding -- (seq_length(time steps), batch, num of features) or (batch, seq_length(time steps),num of features)
            hidden_size=forecast_hidden_size # define the hidden size of the LSTM layer (num of neurons of LSTM hidden layers) -- int, a value
        )

        self.dropout = nn.Dropout(p=output_dropout)

        # Define the connection between hindcast and forecast LSTM -- handoff layers
        self.output_size = output_size
        
        # Hidden state: nonlinear path
        self.hidden_transfer_net = FC(
            input_size=self.hindcast_hidden_size * 2,  # with bilstm
            hidden_sizes=state_handoff_network_para['hiddens'],
            activation=state_handoff_network_para['activation'],
            dropout=state_handoff_network_para['dropout']
        )

        # Cell state: linear path (no activation)
        self.cell_transfer_linear = FC(
            input_size=self.hindcast_hidden_size * 2,  # with bilstm
            hidden_sizes=[self.forecast_hidden_size],
            activation='linear',
            dropout=0.0
        )

        # Hindcast and Forecast head 
        self.forecast_head = get_head(head_type=head_type, n_in=forecast_hidden_size, n_out=self.output_size,output_activation='linear')

        self._reset_parameters()

    def _reset_parameters(self): ###### what does reset-parameter mean??? #####
        """Special initialization of certain model weights."""
        if self.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.hindcast_hidden_size:2 * self.hindcast_hidden_size] = self.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[self.forecast_hidden_size:2 * self.forecast_hidden_size] = self.initial_forget_bias
        # bias_hh_l0 represents the bias term applied to the operations involving the recurrent (hidden-to-hidden) connections for the first layer (layer 0) of the LSTM
        # It affects how the LSTM processes information flowing between timesteps using its recurrent state

    def forward(self, x_glofas_h, x_era5_h, x_cpc, x_input_f,x_static):
        """Perform a forward pass on the MultiheadForecastLSTM model."""
        # Embed hindcast and forecast data
        # concatenate the dynamic inputs: GloFAS, ERA5, CPC
        x_input_h = torch.cat([x_glofas_h, x_era5_h,x_cpc], dim=-1) 
        x_h = self.hindcast_embedding_net(x_input_h, x_static) # (batch,seq,feature)
        x_h = x_h.transpose(0, 1) # (seq,batch,feature) -- swap batch and num_of_layer -- prepare for LSTM batch_first=False
        # x_h from hindcast embedding shape -- (batch, seq, hidden size of embedding*2)

        x_f = self.forecast_embedding_net(x_input_f, x_static) 
        x_f = x_f.transpose(0, 1) # (seq,batch,feature) -- swap batch and num_of_layer

        # shape of x_f: (seq, batch, hidden size of embed layer)

        # hindcast LSTM:
        # input(batch, time, input_size(equal to embedding hidden layer*2)) as the input size for forwarding LSTM
        # Output size from lstm hindcast (seq_len, batch, hidden_size of LSTM * num_of_LSTM layer)
        # shape for h_n_hindcast or c_n_hindcast: (numlayers_LSTM * num_directions(uni or bi), batch, hidden_size)

        _, (h_n_hindcast, c_n_hindcast) = self.hindcast_lstm(x_h) 

        # h_n_hindcast and c_n_hindcast shape will be (2, batch, hidden_size)

        # 动态获取 batch size
        batch_size = h_n_hindcast.shape[1]
        h_n = h_n_hindcast.transpose(0, 1).reshape(batch_size, -1)  # shape (batch, hidden_size*2)
        c_n = c_n_hindcast.transpose(0, 1).reshape(batch_size, -1)  # shape (batch, hidden_size*2)

        # Apply nonlinear transformation to hidden state
        h_n_handoff = self.hidden_transfer_net(h_n).unsqueeze(0)  # shape: (batch, forecast_hidden_size)

        # Apply linear transformation to cell state
        c_n_handoff = self.cell_transfer_linear(c_n).unsqueeze(0)  # shape: (batch, forecast_hidden_size)

        # run forecast lstm
        lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.forecast_lstm(x_f, (h_n_handoff, c_n_handoff))
        # Output size from lstm forecast (seq_len, batch, hidden_size of LSTM * num_of_LSTM layer)

        lstm_output_forecast = lstm_output_forecast.transpose(0, 1) # swap 0 and 1 then get (batch, seq_len, hidden_size of LSTM * num_of_LSTM layer)

        # finally output the prediction through forecast head
        output_forecast = self.forecast_head(self.dropout(lstm_output_forecast)) # shape (batch_size, forecast_seq_length, output_size)

        return output_forecast

#### new run ####

if __name__ == '__main__':
    import torch
    import time
    from torch.cuda.amp import autocast

    # Hyperparameters and dummy shapes
    batch_size = 8
    hindcast_len = 10
    forecast_len = 5
    glofas_features = 4
    era5_features = 6
    cpc_features = 2
    static_features = 10

    # Dummy inputs
    x_glofas_h = torch.randn((batch_size, hindcast_len, glofas_features))
    x_era5_h = torch.randn((batch_size, hindcast_len, era5_features))
    x_cpc = torch.randn((batch_size, hindcast_len, cpc_features))
    x_forecast = torch.randn((batch_size, forecast_len, era5_features))
    x_static = torch.randn((batch_size, static_features))

    # Define model
    model = HandoffForecastLSTM(
        glofas_inputs_size_h=glofas_features,
        era5_inputs_size_h=era5_features,
        inputs_size_f=era5_features,
        cpc_inputs_size=cpc_features,
        static_attributes_size=static_features,
        hindcast_hidden_size=32,
        forecast_hidden_size=64,
        output_dropout=0.1,
        output_size=1,
        initial_forget_bias=3,
        static_embedding_spec={'type': 'fc', 'hiddens': [32], 'activation': 'tanh', 'dropout': 0.0},
        dynamic_embedding_spec={'type': 'fc', 'hiddens': [32], 'activation': 'tanh', 'dropout': 0.0},
        state_handoff_network_para={'hiddens': [64], 'activation': 'tanh', 'dropout': 0.0},
        head_type='regression'
    )

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x_glofas_h = x_glofas_h.to(device)
    x_era5_h = x_era5_h.to(device)
    x_cpc = x_cpc.to(device)
    x_forecast = x_forecast.to(device)
    x_static = x_static.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        for i in range(3):
            start = time.time()
            with autocast():  
                out = model(x_glofas_h, x_era5_h, x_cpc, x_forecast, x_static)
            print(f"Run {i+1}: Output shape: {out.shape} | Time: {time.time() - start:.3f}s")