# ------------------------------------------------------------------
# Script for lstm model inference
# Usually used for the obs-target model, driven by ERA5-Land reanalysis and HRES dataset

# ------------------------------------------------------------------
import torch
import numpy as np
from tqdm import tqdm
import RiverMamba.lstm_repo.utils.utils_edlstm as utils
import time
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from RiverMamba.dataset.LSTM_dataset import GloFAS_Dataset_edlstm
import RiverMamba.lstm_repo.config_lstm.config_lstm_infer_bilstm as config_file
import xarray as xr
from datetime import datetime, timedelta
from RiverMamba.lstm_repo.models_lstm.build_lstm import Model

from datetime import datetime, timedelta


np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# Due to the shifted time in glofas, the final results time steps in the file name should be shifted one day forward as well

# ------------------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def check_dataset_files(dataset):
    missing_files = []
    for file_info in dataset.files:
        for key, file_list in file_info.items():
            for file_path in file_list:
                if not os.path.isfile(file_path):
                    missing_files.append(file_path)
    return missing_files

def inference(config_file):

    utils.fix_seed(0)
    config = config_file.read_arguments(train=True)

    output_path = os.path.join(config.output_path, config.name)
    os.makedirs(output_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # this needs optimization, the way to load original model parameters for training
    # here is only temporal solution
    model = Model(
        data_glofas_size_h=len(config.variables_glofas),
        data_era5_size_h=len(config.variables_era5_land),
        data_size_f=len(config.variables_hres_forecast),
        data_cpc_size=1,
        data_static_size=len(config.variables_static),
        hindcast_hidden_size=config.hindcast_hidden_size,
        forecast_hidden_size=config.forecast_hidden_size,
        output_dropout=config.output_dropout,
        output_size=config.output_size,
        initial_forget_bias=config.initial_forget_bias,
        static_embedding_spec=config.static_embedding_spec,
        dynamic_embedding_spec=config.dynamic_embedding_spec,
        state_handoff_network_para=config.state_handoff_network_para,
        head_type=config.head_type
    )


    # wrap a DataParallel model -- add module prefix to the state_dict keys
    model = torch.nn.DataParallel(model)

    model.to(device)

    # load saved model
    checkpoint_path = os.path.join(config.checkpoint_path, config.name,'model_checkpoints/best_loss_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    #model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # load the dataset

    dataset = GloFAS_Dataset_edlstm(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_static=config.root_static,
        root_obs=config.root_obs,
        root_cpc=config.root_cpc,
        is_hres_forecast=config.is_hres_forecast,
        nan_fill=config.nan_fill,
        delta_t=config.delta_t, # the length of the input -- it generates a continuous time series
        delta_t_f=config.delta_t_f, # forecast length
        is_random_t=config.is_random_t, #False,
        is_aug=False,
        is_shuffle=False,
        variables_glofas=config.variables_glofas,
        variables_era5_land=config.variables_era5_land,
        variables_static=config.variables_static,
        variables_hres_forecast=config.variables_hres_forecast,
        variables_glofas_log1p=config.variables_glofas_log1p, # None
        variables_era5_land_log1p=config.variables_era5_land_log1p, # None
        variables_static_log1p=config.variables_static_log1p,
        variables_hres_forecast_log1p=config.variables_hres_forecast_log1p,
        is_norm=config.is_norm, # True/False
        years=config.years_test, # years_val
        lat_min=config.lat_min, # 30
        lat_max=config.lat_max, # 60
        lon_min=config.lon_min, # -10
        lon_max=config.lon_max # 40
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.n_workers_val)

    print("Starting inference...")

    for i, (data_glofas, data_era5_land, data_forecast, data_static,data_cpc,_,
                _, _,_,file_names) in tqdm(enumerate(dataloader), total=len(dataloader)):

        # the reshape here is necessary, otherwise the model will not work with 4 dim tensors
        batch_size_actual = data_glofas.shape[0]  # Use actual batch size
        data_glofas = data_glofas.view(batch_size_actual*config.n_points,data_glofas.shape[2],-1) # (batch*points, seq, features) 
        data_era5_land = data_era5_land.view(batch_size_actual*config.n_points,data_era5_land.shape[2],-1)
        data_forecast = data_forecast.view(batch_size_actual*config.n_points,data_forecast.shape[2],-1) # this is from hres
        data_cpc = data_cpc.view(batch_size_actual*config.n_points,data_cpc.shape[2],-1)
        data_static = data_static.view(batch_size_actual*config.n_points,-1)

        data_glofas = data_glofas.to(torch.float32).to(device)
        data_era5_land = data_era5_land.to(torch.float32).to(device)
        data_forecast = data_forecast.to(torch.float32).to(device)
        data_static = data_static.to(torch.float32).to(device)
        data_cpc = data_cpc.to(torch.float32).to(device)

        file_name = file_names[0]

        with torch.no_grad():
            pred, _ = model(data_glofas, data_era5_land, data_cpc, data_forecast, data_static)

        # log inverse transform
        pred = pred.cpu().numpy().astype(np.float64)
        
        pred = dataset.log1p_inv_transform(pred)

        # extract file name for NetCDF based on main date file -- now we have the file names -- change this lines
        
        pred = pred.squeeze(-1)           # pred.shape == (3366, 7)
        pred = pred.transpose(1, 0)         # pred.shape == (7, 3366)

        # prepare the baseline GloFAS data

        if isinstance(data_glofas, np.ndarray):
            data_glofas = torch.tensor(data_glofas, dtype=torch.float32, device='cpu')

        if config.is_norm:
            data_glofas = data_glofas.cpu()
                   
            data_glofas = dataset.inv_transform(data_glofas, dataset.glofas_mean, dataset.glofas_std)

        data_glofas = data_glofas[:, -1, 1:1+1].detach().numpy()

        data_glofas = data_glofas.transpose(1, 0)         # data_glofas.shape == (1, 3366)

        # from increment to absolute value
        pred_convert = pred + data_glofas

        print (f"Pred shape: {pred.shape}")

        print (f"data_glofas shape: {data_glofas.shape}")

        print (f"pred_convert shape: {pred_convert.shape}")

        output_file = os.path.join(config.output_path, config.name ,f"{file_name}.nc")
        data_out = xr.Dataset(data_vars=dict(dis24=(["step", "x"], pred_convert.astype(np.float32))))
        data_out.to_netcdf(output_file)
        print(f"Saved prediction to {output_file}")

    print("Inference completed!")


if __name__ == '__main__':

    inference(config_file)

