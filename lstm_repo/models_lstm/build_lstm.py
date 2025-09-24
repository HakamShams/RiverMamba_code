# ------------------------------------------------------------
import torch
import torch.nn as nn
import importlib

from RiverMamba.lstm_repo.models_lstm.handoff_lstm import HandoffForecastLSTM
from RiverMamba.lstm_repo.models_lstm.losses import MSE, L1
import numpy as np

import torch.nn.functional as F
import torch.autograd as autograd

# ------------------------------------------------------------
class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class Model(nn.Module): 

    """
    Full model wrapper for handoff edLSTM with STE-based log1p transformation and loss computation.
    Includes:
        - HandoffForecastLSTM as backbone
        - StraightThroughEstimator for sign-preserving log1p transformation
        - MSE and L1 loss functions
    """

    def __init__(self, 
                 data_glofas_size_h, 
                 data_era5_size_h, 
                 data_size_f,
                 data_cpc_size,
                 data_static_size,
                 hindcast_hidden_size: int, # define the hindcast hidden network neurons
                 forecast_hidden_size: int, # define the forecast hidden network neurons
                 output_dropout: float, 
                 output_size: int, # target variables number
                 initial_forget_bias,
                 static_embedding_spec,
                 dynamic_embedding_spec,
                 state_handoff_network_para, 
                 head_type: str):

        super(Model, self).__init__()

        """
        Parameters
        ----------
        config : argparse
            configuration file from config.py
        """

        self.handoff_edlstm = HandoffForecastLSTM(
        data_glofas_size_h, 
        data_era5_size_h, 
        data_size_f,
        data_cpc_size,
        data_static_size,
        hindcast_hidden_size, # define the hindcast hidden network neurons
        forecast_hidden_size, # define the forecast hidden network neurons
        output_dropout=output_dropout,
        output_size=output_size,
        initial_forget_bias=initial_forget_bias,
        static_embedding_spec=static_embedding_spec,
        dynamic_embedding_spec=dynamic_embedding_spec,
        state_handoff_network_para=state_handoff_network_para,
        head_type=head_type)

        self.init_weights() # why and how?
        self.ste = StraightThroughEstimator()

        # define the losses
        self.mse = MSE()
        self.l1 = L1()

    def init_weights(self): # using normal distribution as initialization
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # He (Kaiming) initialization for linear layers
                y = m.in_features
                m.weight.data.normal_(0.0, 1/np.sqrt(y)) 
                m.bias.data.fill_(0)
            elif isinstance(m, nn.LSTM):
                # Initialization for recurrent layers
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

        self.apply(_init_weights)

    def log1p_transform(self, x):
        return self.ste(x) * torch.log1p(torch.abs(x))

    def log1p_inv_transform(self, x):
        return self.ste(x) * torch.expm1(torch.abs(x))

    def transform(self, x, mean, std):
        x_norm = (x - mean) / (std + 1e-6)
        return x_norm

    def inv_transform(self, x_norm, mean, std):
        x = (x_norm * (std + 1e-6)) + mean
        return x


    def forward(self, x_glofas_h,x_era5_land_h,x_cpc, x_input_f,x_static,
                y_glofas=None, y_thresholds=None,is_obs=False):
        """
        Forward pass of the full model.

        Parameters
        ----------
        x_glofas_h : torch.Tensor
            Hindcast GloFAS input [B, T, P, C]
        x_era5_land_h : torch.Tensor
            Hindcast ERA5 input
        x_cpc : torch.Tensor
            CPC input (optional)
        x_input_f : torch.Tensor
            Forecast input (no GloFAS)
        x_static : torch.Tensor
            Static input
        y_glofas : torch.Tensor, optional
            Target GloFAS values
        y_thresholds : torch.Tensor, optional
            Thresholds used for flood severity weighting
        is_obs : bool
            Whether targets are observational

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor or None)
            Predicted values and optionally the computed loss
        """

        x = self.handoff_edlstm(x_glofas_h,x_era5_land_h,x_cpc,x_input_f,x_static)

        if y_glofas is not None:
            if is_obs:

                mask = ~torch.isnan(y_glofas)  # shape: same as y_glofas
                valid_count = mask.sum()

                if valid_count == 0:
                    loss_mse = torch.tensor(0.0, device=x.device)
                else:
                    x_filtered = x[mask]
                    y_filtered = y_glofas[mask]
                    loss_mse = self.mse(x_filtered, y_filtered)

            else:

                loss_mse = self.mse(x, y_glofas)

            if y_thresholds is not None: # could be 
                y_thresholds = torch.sign(y_thresholds) * torch.log1p(torch.abs(y_thresholds))
                y_thresholds = y_thresholds - (
                            (x_glofas[:, -1, :, 1:1 + 1] * (1.2420777850270224 + 1e-6)) + 0.5858954953406135)

                loss_cls = self.l1(x, y_glofas, y_thresholds)
            else:
                loss_cls = 0

            loss = (loss_mse + loss_cls).unsqueeze(0)
        else:
            loss = None

        return (x, loss)

if __name__ == '__main__':

    import os

    device = 'cpu'


    model = Model(
        data_glofas_size_h=4, 
        data_era5_size_h=24,
        data_size_f=7,
        data_cpc_size=10,
        data_static_size=299,
        hindcast_hidden_size=128,
        forecast_hidden_size=128,
        output_dropout=0.1,
        output_size=1,
        initial_forget_bias=3,
        static_embedding_spec={'type': 'fc', 'hiddens': [[30, 20, 64]], 'activation': 'tanh', 'dropout': 0},
        dynamic_embedding_spec={'type': 'fc', 'hiddens': [[30, 20, 64]], 'activation': 'tanh', 'dropout': 0},
        state_handoff_network_para={'hiddens': [256, 128], 'activation': 'tanh', 'dropout': 0.1},
        head_type='regression'
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    import time

    start = time.time()
    for epoch in range(2):
        optimizer.zero_grad()

        data_glofas = torch.randn((64, 15, 4))
        data_era5_land = torch.randn((64, 15, 24))
        data_cpc = torch.randn((64, 15, 10))
        data_input_f = torch.randn((64, 5, 7)) # forecast input
        data_static = torch.randn((64, 299))
        data_glofas_target = torch.randn((64, 5, 1))

        output, loss = model(data_glofas, data_era5_land, data_cpc, data_input_f, data_static, data_glofas_target)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print(f"output shape is {output.shape}")
        print(f"loss shape is {loss.shape}")

        loss.backward()
        optimizer.step()

    print("Training time:", time.time() - start)

    model.eval()
    with torch.no_grad():
        for epoch in range(2):
            data_glofas = torch.randn((64, 15, 4))
            data_era5_land = torch.randn((64, 15, 24))
            data_cpc = torch.randn((64, 15, 10))
            data_input_f = torch.randn((64, 5, 7))
            data_static = torch.randn((64, 299))
            data_glofas_target = torch.randn((64, 5, 1))

            output, loss = model(data_glofas, data_era5_land, data_cpc, data_input_f, data_static, data_glofas_target)
            print(f"Evaluation Epoch {epoch + 1}, Loss: {loss.item()}")
            print(f"output shape is {output.shape}")
            print(f"loss shape is {loss.shape}")
