# ------------------------------------------------------------------
"""
Script for testing and validating RiverMamba on full resolution map

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
import utils.utils as utils
from models.build import Model
import time
import os
from torch.utils.tensorboard import SummaryWriter
from dataset.RiverMamba_dataset import RiverMamba_Dataset
import config as config_file
from torch import autocast
import xarray as xr

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# ------------------------------------------------------------------

output_path = r'./inference_full_map'
model_path = r'./RiverMamba_full_map_reanalysis.pth'

os.makedirs(output_path, exist_ok=True)

batch_size = 1
num_workers = 0
name = 'inference_full_map'
years_test = ['2019', '2020', '2021', '2022', '2023', '2024']

is_obs = False
n_points = 311097

sample_division = int(np.ceil(6221926/n_points))

is_autocast = False

def test(config_file):
    # read config arguments
    config = config_file.read_arguments(train=False, print=True, save=False)

    config.pretrained_model = model_path
    config.name = name
    config.is_hres_forecast = True
    config.is_obs = is_obs
    config.n_points = n_points
    config.years_test = years_test
    config.en_use_checkpoint = False
    config.num_workers = num_workers

    config.en_grouping_size = [(4, n_points), (2, n_points), (1, n_points)]
    config.de_grouping_size = [(1, n_points)]

    # get logger
    logger = utils.get_logger(config)

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading validation dataset ...")

    val_dataset = RiverMamba_Dataset(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_static=config.root_static,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_cpc=config.root_cpc,
        root_obs=config.root_obs,
        nan_fill=config.nan_fill,
        delta_t=config.delta_t,
        delta_t_f=config.delta_t_f,
        is_hres_forecast=True,
        is_shuffle=False,
        is_sample_aifas=False,
        is_sample=False,
        n_points=6221926,
        variables_glofas=config.variables_glofas,
        variables_era5_land=config.variables_era5_land,
        variables_static=config.variables_static,
        variables_hres_forecast=config.variables_hres_forecast,
        variables_cpc=config.variables_cpc,
        variables_glofas_log1p=config.variables_glofas_log1p,
        variables_era5_land_log1p=config.variables_era5_land_log1p,
        variables_static_log1p=config.variables_static_log1p,
        variables_hres_forecast_log1p=config.variables_hres_forecast_log1p,
        variables_cpc_log1p=config.variables_cpc_log1p,
        is_add_xyz=config.is_add_xyz,
        curves=config.curves,
        is_shuffle_curves=False,
        is_norm=config.is_norm,
        years=config.years_val,
        lat_min=None,  # config.lat_min,
        lat_max=None,  # config.lat_max,
        lon_min=None,  # config.lon_min,
        lon_max=None,  # config.lon_max
        static_dataset=config.static_dataset,
        is_sample_curves=False,
        is_obs=False,
        is_val=True
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=config.pin_memory,
                                                 num_workers=config.num_workers
                                                 )

    utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    if config.gpu_id != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    model = Model(config)

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    utils.log_string(logger, "decoder parameters: %d" % utils.count_parameters(model.decoder))
    utils.log_string(logger, "regression head parameters: %d" % utils.count_parameters(model.heads))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    utils.log_string(logger, 'inference from RiverMamba ...\n')
    time.sleep(1)

    # validation
    with torch.no_grad():

        model.eval()

        time.sleep(1)

        pbar = tqdm(val_dataloader, total=len(val_dataloader), smoothing=0.9, postfix="  training")

        for data in pbar:

            data_glofas_i = data['glofas']  # B, T, P, C
            data_era5_land_i = data['era5']
            data_static_i = data['static']
            data_hres_i = data['hres_forecast']
            data_cpc_i = data['cpc']
            curves_i = data['curves']
            file_name = data['file_name']

            preds = np.zeros((len(file_name), config.delta_t_f, data_glofas_i.shape[2], 1))
            preds[:] = np.nan

            curves_i = curves_i.cpu().numpy()[0]  # number of curves, 2, 2

            # divide the data along the first curve and run inference sequentially
            for r in range(sample_division):

                random_indices = curves_i[2, 0, :][n_points * r: n_points * (r + 1)]
                #random_indices = val_dataset.curves_series[0][r::sample_division][:n_points]
                #random_indices = curves_i[2, 0, :][r::sample_division]

                sampled_series = np.arange(len(random_indices))
                dictionary = dict(zip(random_indices, sampled_series))
                curves = []
                for curve, _ in curves_i:
                    curve_j = list(map(dictionary.get, curve[np.isin(curve, random_indices, assume_unique=True)]))
                    dictionary_j = dict(zip(curve_j, sampled_series))
                    curves.append([curve_j, list(map(dictionary_j.get, sampled_series))])
                curves = torch.tensor(np.array(curves).astype(np.int32))[None, :, :, :].repeat(batch_size, 1, 1, 1)

                data_glofas = data_glofas_i[:, :, random_indices, :]
                data_era5_land = data_era5_land_i[:, :, random_indices, :]
                data_static = data_static_i[:, random_indices, :]
                data_hres = data_hres_i[:, :, random_indices, :]
                data_cpc = data_cpc_i[:, :, random_indices, :]

                if is_autocast:
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        pred = model(data_hres.to(device),
                                     data_glofas.to(device),
                                     data_era5_land.to(device),
                                     data_cpc.to(device),
                                     data_static.to(device),
                                     curves.to(device).long(),
                                     None,
                                     None,
                                     None
                                     )
                else:
                    pred = model(data_hres.to(device),
                                 data_glofas.to(device),
                                 data_era5_land.to(device),
                                 data_cpc.to(device),
                                 data_static.to(device),
                                 curves.to(device).long(),
                                 None,
                                 None,
                                 None
                                 )
                
                preds[:, :, random_indices, :] = pred.cpu().float().numpy().astype(np.float64)

            preds = val_dataset.log1p_inv_transform(preds)
            
            data_glofas_i = data_glofas_i[:, -1:, :, 1:1 + 1].cpu().numpy()
            data_glofas_i = val_dataset.inv_transform(data_glofas_i, 53.86304994405949, 1621.447677699888)
            #data_glofas_i = val_dataset.inv_transform(data_glofas_i,
            #                                          val_dataset.glofas_mean[val_dataset.dis24_index],
            #                                          val_dataset.glofas_std[val_dataset.dis24_index]
            #                                          )

            preds = data_glofas_i + preds
            preds = np.clip(preds, 0, a_max=None)

            preds = preds[:, :, :, 0]

            for b in range(len(preds)):

                file_output = os.path.join(output_path, file_name[b] + '.nc')

                data_out = xr.Dataset(data_vars=dict(dis24=(["time", "x"], preds[b].astype(np.float32))))
                data_out.to_netcdf(file_output)


if __name__ == '__main__':

    test(config_file)

