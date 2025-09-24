# ------------------------------------------------------------
import torch
import torch.nn as nn
import importlib
#from timm.models.layers import trunc_normal_
from RiverMamba.lstm_repo.models_lstm.handoff_bilstm import HandoffForecastLSTM
from RiverMamba.lstm_repo.models_lstm.losses import MSE, L1
import numpy as np

import torch.nn.functional as F
import torch.autograd as autograd

# ------------------------------------------------------------
# Straight-Through Estimator (STE) for sign-preserving transforms
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
    """Applies sign-preserving STE operation for log1p-style transforms."""
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class Model(nn.Module): 
    """
    A wrapper model that encapsulates the HandoffForecastLSTM model, including:
    - Custom weight initialization
    - Log1p transform utilities with signed STE
    - Loss computation with optional weighted loss and classification loss (threshold-based)

    Parameters
    ----------
    data_glofas_size_h : int
        Number of GloFAS features for hindcast.
    data_era5_size_h : int
        Number of ERA5 features for hindcast.
    data_size_f : int
        Number of dynamic features for forecast stage.
    data_cpc_size : int
        Number of CPC hindcast inputs.
    data_static_size : int
        Number of static features.
    hindcast_hidden_size : int
        Hidden size of BiLSTM used for hindcast encoding.
    forecast_hidden_size : int
        Hidden size of LSTM used for forecast decoding.
    output_dropout : float
        Dropout rate applied to the forecast head output.
    output_size : int
        Number of target variables to forecast.
    initial_forget_bias : float
        Initial forget gate bias for LSTM layers.
    static_embedding_spec : dict
        Specification for static input embedding (FC architecture).
    dynamic_embedding_spec : dict
        Specification for dynamic input embedding (FC architecture).
    state_handoff_network_para : dict
        Specification for hidden/cell state transformation between BiLSTM and forecast LSTM.
    head_type : str
        Type of regression head used for forecasting (e.g., 'regression', 'gmm').
    use_weighted_loss : bool, optional
        Whether to apply severity-weighted MSE loss. Default: True.
    """

    def __init__(self, 
                 data_glofas_size_h, 
                 data_era5_size_h, 
                 data_size_f,
                 data_cpc_size,
                 data_static_size,
                 hindcast_hidden_size: int,
                 forecast_hidden_size: int,
                 output_dropout: float, 
                 output_size: int,
                 initial_forget_bias,
                 static_embedding_spec,
                 dynamic_embedding_spec,
                 state_handoff_network_para, 
                 head_type: str,
                 use_weighted_loss: bool = True):

        super(Model, self).__init__()

        self.use_weighted_loss = use_weighted_loss

        self.handoff_edlstm = HandoffForecastLSTM(
            data_glofas_size_h, 
            data_era5_size_h, 
            data_size_f,
            data_cpc_size,
            data_static_size,
            hindcast_hidden_size,
            forecast_hidden_size,
            output_dropout,
            output_size,
            initial_forget_bias,
            static_embedding_spec,
            dynamic_embedding_spec,
            state_handoff_network_para,
            head_type
        )

        self.init_weights()
        self.ste = StraightThroughEstimator()

        self.mse = MSE()
        self.l1 = L1()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                y = m.in_features
                m.weight.data.normal_(0.0, 1 / np.sqrt(y)) 
                m.bias.data.fill_(0)
            elif isinstance(m, nn.LSTM):
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
        return (x - mean) / (std + 1e-6)

    def inv_transform(self, x_norm, mean, std):
        return (x_norm * (std + 1e-6)) + mean

    def forward(self, x_glofas_h, x_era5_land_h, x_cpc, x_input_f, x_static,
                y_glofas=None, y_thresholds=None, weight_map_thr=None, is_obs=False):

        """
        Run the forward pass and optionally compute loss if labels are provided.

        Parameters
        ----------
        x_glofas_h : Tensor
            GloFAS hindcast inputs (B, T, C).
        x_era5_land_h : Tensor
            ERA5 hindcast inputs (B, T, C).
        x_cpc : Tensor
            CPC hindcast inputs (B, T, C).
        x_input_f : Tensor
            Forecast inputs (B, T, C).
        x_static : Tensor
            Static input tensor (B, C_static).
        y_glofas : Tensor, optional
            Forecast target values (B, T, output_size).
        y_thresholds : Tensor, optional
            Threshold levels (for classification loss).
        weight_map_thr : Tensor, optional
            Severity weight map (B, T, 1).
        is_obs : bool
            Whether y_glofas comes from observations.

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Forecast output (B, T, output_size), and loss if target is provided.
        """

        x = self.handoff_edlstm(x_glofas_h, x_era5_land_h, x_cpc, x_input_f, x_static)


        if y_glofas is not None:
            if is_obs:
                mask = ~torch.isnan(y_glofas).squeeze(-1)  # (B, F)

                valid_count = mask.sum()

                if valid_count == 0:
                    loss_mse = torch.tensor(0.0, device=x.device)
                else:
                    x_filtered = x[mask]
                    y_filtered = y_glofas[mask]
                    if self.use_weighted_loss and weight_map_thr is not None:
                        weights_filtered = weight_map_thr[mask].unsqueeze(-1)
                    else:
                        weights_filtered = None
                    loss_mse = self.mse(x_filtered, y_filtered, weights_filtered)
            else:
                weights = weight_map_thr if self.use_weighted_loss else None
                loss_mse = self.mse(x, y_glofas, weights)

            if y_thresholds is not None:
                y_thresholds = torch.sign(y_thresholds) * torch.log1p(torch.abs(y_thresholds))
                y_thresholds = y_thresholds - (
                    (x_glofas_h[:, -1, :, 1:2] * (1.2420777850270224 + 1e-6)) + 0.5858954953406135)
                loss_cls = self.l1(x, y_glofas, y_thresholds)
            else:
                loss_cls = 0.0

            loss = (loss_mse + loss_cls).unsqueeze(0)
        else:
            loss = None

        return (x, loss)

if __name__ == '__main__':
    import torch
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        static_embedding_spec={'type': 'fc', 'hiddens': [30, 20, 64], 'activation': 'tanh', 'dropout': 0},
        dynamic_embedding_spec={'type': 'fc', 'hiddens': [30, 20, 64], 'activation': 'tanh', 'dropout': 0},
        state_handoff_network_para={'hiddens': [256, 128], 'activation': 'tanh', 'dropout': 0.1},
        head_type='regression'
    ).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    start = time.time()
    for epoch in range(2):
        model.train()
        optimizer.zero_grad()

        data_glofas = torch.randn((64, 15, 4), device=device)
        data_era5_land = torch.randn((64, 15, 24), device=device)
        data_cpc = torch.randn((64, 15, 10), device=device)
        data_input_f = torch.randn((64, 5, 7), device=device)
        data_static = torch.randn((64, 299), device=device)
        data_glofas_target = torch.randn((64, 5, 1), device=device)

        weight_map_thr = torch.rand((64, 5), device=device)  # (batch_size, forecast_len)

        output, loss = model(
            data_glofas, data_era5_land, data_cpc, data_input_f, data_static,
            data_glofas_target, weight_map_thr=weight_map_thr
        )

        print(f"Train Epoch {epoch + 1}, Loss: {loss.item()}")
        print(f"Output shape: {output.shape}, Loss shape: {loss.shape}")

        loss.backward()
        optimizer.step()

    print(f"Training time: {time.time() - start:.2f} seconds")

    ###### evaluation ######
    model.eval()
    with torch.no_grad():
        for epoch in range(2):
            data_glofas = torch.randn((64, 15, 4), device=device)
            data_era5_land = torch.randn((64, 15, 24), device=device)
            data_cpc = torch.randn((64, 15, 10), device=device)
            data_input_f = torch.randn((64, 5, 7), device=device)
            data_static = torch.randn((64, 299), device=device)
            data_glofas_target = torch.randn((64, 5, 1), device=device)

            weight_map_thr = torch.rand((64, 5), device=device)

            output, loss = model(
                data_glofas, data_era5_land, data_cpc, data_input_f, data_static,
                data_glofas_target, weight_map_thr=weight_map_thr
            )

            print(f"Eval Epoch {epoch + 1}, Loss: {loss.item()}")
            print(f"Output shape: {output.shape}, Loss shape: {loss.shape}")
