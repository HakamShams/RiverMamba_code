# ------------------------------------------------------------------
"""
Script for training and validating RiverMamba

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
from torch.cuda.amp import GradScaler
from torch import autocast

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# torch.autograd.set_detect_anomaly(True)

# ------------------------------------------------------------------


def train():
    # read config arguments
    config = config_file.read_arguments(train=True)

    # get logger
    logger = utils.get_logger(config)

    # get tensorboard writer
    writer = SummaryWriter(os.path.join(config.dir_log, config.name))

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading training dataset ...")

    train_dataset = RiverMamba_Dataset(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_static=config.root_static,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_cpc=config.root_cpc,
        root_obs=config.root_obs,
        nan_fill=config.nan_fill,
        delta_t=config.delta_t,
        delta_t_f=config.delta_t_f,
        is_hres_forecast=config.is_hres_forecast,
        is_shuffle=config.is_shuffle,
        is_sample=config.is_sample,
        is_sample_aifas=config.is_sample_aifas,
        n_points=config.n_points,
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
        is_shuffle_curves=config.is_shuffle_curves,
        is_norm=config.is_norm,
        years=config.years_train,
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lon_min=config.lon_min,
        lon_max=config.lon_max,
        is_obs=config.is_obs,
        alpha=config.alpha,
        static_dataset=config.static_dataset,
        is_sample_curves=config.is_sample_curves,
        is_val=False
    )

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
        is_hres_forecast=config.is_hres_forecast,
        is_shuffle=False,
        is_sample=config.is_sample,
        is_sample_aifas=config.is_sample_aifas,
        n_points=config.n_points,
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
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lon_min=config.lon_min,
        lon_max=config.lon_max,
        is_obs=config.is_obs,
        alpha=config.alpha,
        static_dataset=config.static_dataset,
        is_sample_curves=config.is_sample_curves,
        is_val=True
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size_train,
                                                   shuffle=True,
                                                   pin_memory=config.pin_memory,
                                                   num_workers=config.n_workers_train)

    # random_sampler = torch.utils.data.RandomSampler(val_dataset, num_samples=20*len(val_dataset))

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config.batch_size_val,
                                                 drop_last=False,
                                                 # sampler=random_sampler,
                                                 shuffle=True,
                                                 pin_memory=config.pin_memory,
                                                 num_workers=config.n_workers_val)

    utils.log_string(logger, "# training samples: %d" % len(train_dataset))
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

    # get optimizer and scheduler
    utils.log_string(logger, "get optimizer and learning rate scheduler ...")

    optimizer = utils.get_optimizer(model, config)
    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    # training loop
    utils.log_string(logger, 'training on GloFAS dataset ...\n')

    eval_val = utils.evaluator(logger, 'Validation', lead_time=train_dataset.delta_t_f)

    time.sleep(1)

    # initialize the best values
    best_loss_train, best_loss_val = np.inf, np.inf
    best_F1_val, best_R2_val = 0, 0

    scaler = GradScaler()
    max_norm = config.max_norm
    is_obs = config.is_obs

    if config.training_checkpoint:
        checkpoint = torch.load(config.training_checkpoint)
        epoch_checkpoint = checkpoint['epoch'] + 1
        best_loss_train = checkpoint['mean_loss_train']
        best_loss_val = checkpoint['best_loss_validation']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('resume training from epoch {} with best val loss {}'.format(epoch_checkpoint, best_loss_val))
        del checkpoint
    else:
        epoch_checkpoint = 0

    for epoch in range(epoch_checkpoint, config.n_epochs):
        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config.n_epochs))

        lr_scheduler.step_update(epoch)

        # training
        model.train()
        loss_train = 0
        time.sleep(1)

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9, postfix="  training")

        for i, data_i in pbar:

            optimizer.zero_grad(set_to_none=True)

            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                pred, loss = model(data_i['hres_forecast'].to(device),
                                   data_i['glofas'].to(device),
                                   data_i['era5'].to(device),
                                   data_i['cpc'].to(device),
                                   data_i['static'].to(device),
                                   data_i['curves'].to(device).long(),
                                   data_i['glofas_target'].to(device) if not is_obs else None,
                                   data_i['obs_target'].to(device) if is_obs else None,
                                   data_i['weight'].to(device),
                                   )

                loss = torch.nan_to_num(loss)

                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()

            pbar.set_description('loss %s' % (loss.item()), refresh=True)

            loss_train += loss.item()

        mean_loss_train = loss_train / float(len(train_dataloader))

        utils.log_string(logger, '%s mean loss     : %.6f' % ('Training', mean_loss_train))
        utils.log_string(logger, '%s best mean loss: %.6f\n' % ('Training', best_loss_train))

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train

        utils.save_model(model, optimizer, epoch, mean_loss_train, np.nan,
                         best_loss_train, best_loss_val, logger, config, 'train')

        # validation
        if epoch % config.val_every_n_epochs == 0:

            with torch.no_grad():

                model.eval()
                loss_val = 0

                time.sleep(1)

                for i, data_i in tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                                      smoothing=0.9,  postfix="  validation"):

                    # should be  # B, T, P, V
                    data_glofas = data_i['glofas']
                    data_era5_land = data_i['era5']
                    data_static = data_i['static']
                    data_weight = data_i['weight']
                    data_hres = data_i['hres_forecast']
                    data_cpc = data_i['cpc']
                    curves = data_i['curves']
                    target = data_i['glofas_target'] if not is_obs else data_i['obs_target']

                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        pred, loss = model(data_hres.to(device),
                                           data_glofas.to(device),
                                           data_era5_land.to(device),
                                           data_cpc.to(device),
                                           data_static.to(device),
                                           curves.to(device).long(),
                                           target.to(device) if not is_obs else None,
                                           target.to(device) if is_obs else None,
                                           data_weight.to(device),
                                           )

                    loss_val += loss.item()

                    target = target.numpy().astype(np.float64)
                    pred = pred.cpu().float().numpy().astype(np.float64)

                    pred = train_dataset.log1p_inv_transform(pred)
                    target = train_dataset.log1p_inv_transform(target)

                    data_glofas = data_glofas[:, -1, :, val_dataset.dis24_index:val_dataset.dis24_index + 1].cpu().numpy()
                    data_glofas = train_dataset.inv_transform(data_glofas,
                                                              train_dataset.glofas_mean[val_dataset.dis24_index],
                                                              train_dataset.glofas_std[val_dataset.dis24_index]
                                                              )

                    data_thresholds = data_i['thresholds'].cpu().numpy()
                    data_lead_time = np.repeat(np.arange(1, config.delta_t_f + 1)[None, :], len(pred), axis=0)

                    for t in range(config.delta_t_f):
                        pred_t = pred[:, t, :, :] + data_glofas
                        target_t = target[:, t, :, :] + data_glofas

                        pred_t = np.clip(pred_t, 0, a_max=None)
                        target_t = np.clip(target_t, 0, a_max=None)

                        # TODO evaluate w.r.t. obs
                        eval_val(pred_t,
                                 target_t,
                                 data_thresholds,
                                 data_lead_time[:, t]
                                 )

                mean_loss_val = loss_val / float(len(val_dataloader))
                eval_val.get_results()

                utils.log_string(logger, '%s mean loss     : %.6f' % ('Validation', mean_loss_val))
                utils.log_string(logger, '%s best mean loss: %.6f\n' % ('Validation', best_loss_val))

                if mean_loss_val <= best_loss_val:
                    best_loss_val = mean_loss_val
                    utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val,
                                     best_loss_train, best_loss_val, logger, config, 'loss')

                if eval_val.F1 >= best_F1_val:
                    best_F1_val = eval_val.F1
                    # utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'F1')

                if eval_val.dis_r2 >= best_R2_val:
                    best_R2_val = eval_val.dis_r2
                    # utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'R2')

            writer.add_scalars("Loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch + 1)

        eval_val.reset()


if __name__ == '__main__':
    train()

