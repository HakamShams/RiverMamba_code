# ------------------------------------------------------------------
"""
Script for training and validating RiverMamba on multinode GPUs

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
import utils.utils as utils
from models.build import Model
import time
import os
from torch.utils.tensorboard import SummaryWriter
from dataset.RiverMamba_dataset import RiverMamba_Dataset
import config as config_file
from torch.amp import GradScaler
from torch import autocast
from socket import gethostname
from datetime import timedelta

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader as DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging

# torch.autograd.set_detect_anomaly(True)

# ------------------------------------------------------------------


def ddp_setup(local_rank):
    torch.cuda.set_device(local_rank)
    timeout_long_ncll = timedelta(seconds=6000)  # 100 minutes
    torch.distributed.init_process_group(backend="nccl", timeout=timeout_long_ncll)


def prepare_dataloader(dataset, batch_size=4, pin_memory=False, num_workers=0, drop_last=False, seed=0):
    sampler = DistributedSampler(dataset, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=drop_last, sampler=sampler)
    return dataloader


def main():
    if "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        #local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        #local_rank = int(os.environ["LOCAL_RANK"])

    # world_size = int(os.environ["WORLD_SIZE"])
    # rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # setup the process groups
    ddp_setup(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

    is_print = (local_rank == 0 and rank == 0)

    # read config arguments
    config = config_file.read_arguments(train=True, print=False)

    if is_print:
        # get logger
        logger = utils.get_logger(config)
        # get tensorboard writer
        writer = SummaryWriter(os.path.join(config.dir_log, config.name))

    if is_print:
        utils.log_string(logger, "SLURM_NTASKS: %i" % int(os.environ["SLURM_LOCALID"]))
        utils.log_string(logger, "SLURM_CPUS_PER_TASK: %i" % int(os.environ["SLURM_CPUS_PER_TASK"]))
        utils.log_string(logger, "SLURM_GPUS_ON_NODE: %i" % int(os.environ["SLURM_GPUS_ON_NODE"]))
        utils.log_string(logger, "SLURM_NNODES: %i" % int(os.environ["SLURM_NNODES"]))

    # fix random seed
    utils.fix_seed(config.seed)

    # dataset
    if is_print: utils.log_string(logger, "loading training dataset ...")
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
        is_val=False,
    )

    if is_print: utils.log_string(logger, "loading validation dataset ...")
    val_dataset = RiverMamba_Dataset(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_static=config.root_static,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_obs=config.root_obs,
        root_cpc=config.root_cpc,
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
        is_val=True,
    )

    # prepare the dataloader
    train_dataloader = prepare_dataloader(train_dataset,
                                          batch_size=config.batch_size_train,
                                          pin_memory=config.pin_memory,
                                          num_workers=config.n_workers_train,
                                          drop_last=True,
                                          seed=config.seed)

    val_dataloader = prepare_dataloader(val_dataset,
                                        batch_size=config.batch_size_val,
                                        pin_memory=config.pin_memory,
                                        num_workers=config.n_workers_val,
                                        drop_last=False,
                                        seed=config.seed)

    if is_print: utils.log_string(logger, "# training samples: %d" % len(train_dataset))
    if is_print: utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))

    time.sleep(4)

    # get models
    if is_print: utils.log_string(logger, "\nloading the model ...")

    model = Model(config)

    if is_print: utils.log_string(logger, "model parameters ...")
    if is_print: utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    if is_print: utils.log_string(logger, "decoder parameters: %d" % utils.count_parameters(model.decoder))
    if is_print: utils.log_string(logger, "regression head parameters: %d" % utils.count_parameters(model.heads))
    if is_print: utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # get optimizer
    if is_print: utils.log_string(logger, "get optimizer and learning rate scheduler ...")

    optimizer = utils.get_optimizer(model, config)

    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    model = DDP(model.to(local_rank), device_ids=[local_rank],
                output_device=local_rank)  # , find_unused_parameters=True)

    # training
    if is_print: utils.log_string(logger, 'training on GloFAS dataset ...\n')

    scaler = GradScaler('cuda')

    # initialize the best values
    best_loss_train, best_loss_val = np.inf, np.inf

    n = len(train_dataloader) * int(os.environ["SLURM_GPUS_ON_NODE"]) * int(os.environ["SLURM_NNODES"])
    n_val = len(val_dataloader) * int(os.environ["SLURM_GPUS_ON_NODE"]) * int(os.environ["SLURM_NNODES"])

    is_obs = config.is_obs
    max_norm = config.max_norm

    # load checkpoint if exists
    if config.training_checkpoint:
        checkpoint = torch.load(config.training_checkpoint)
        epoch_checkpoint = checkpoint['epoch'] + 1
        best_loss_train = checkpoint['mean_loss_train']
        best_loss_val = checkpoint['best_loss_validation']
        optimizer.load_state_dict(checkpoint['optimizer'])
        if is_print: utils.log_string(logger,
                                      'resume training from epoch {} with best val loss {}'.format(epoch_checkpoint,
                                                                                                   best_loss_val))
        del checkpoint
    else:
        epoch_checkpoint = 0

    for epoch in range(epoch_checkpoint, config.n_epochs):

        if is_print:
            utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config.n_epochs))

        lr_scheduler.step_update(epoch)

        _, loss_train = train(model, local_rank, train_dataloader, scaler, optimizer, epoch, is_print, is_obs, max_norm)

        mean_loss_train = reduce_tensor(torch.tensor(loss_train).to(local_rank).detach()).item() / n

        if is_print: utils.log_string(logger, '%s mean loss     : %.6f' % ('Training', mean_loss_train))
        if is_print: utils.log_string(logger, '%s best mean loss: %.6f\n' % ('Training', best_loss_train))

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train

        # if is_print: utils.save_model(model, optimizer, epoch, mean_loss_train, np.nan, best_loss_train, best_loss_val, logger, config, 'train')

        _, loss_val = val(model, local_rank, val_dataloader, epoch, is_print, is_obs)

        mean_loss_val = reduce_tensor(torch.tensor(loss_val).to(local_rank)).item() / n_val

        if is_print: utils.log_string(logger, '%s mean loss     : %.6f' % ('Validation', mean_loss_val))
        if is_print: utils.log_string(logger, '%s best mean loss: %.6f\n' % ('Validation', best_loss_val))

        if mean_loss_val <= best_loss_val:
            best_loss_val = mean_loss_val
            if is_print:
                utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, best_loss_train,
                                 best_loss_val, logger, config, 'loss')

        if is_print: utils.save_model(model, optimizer, epoch, mean_loss_train, np.nan, best_loss_train, best_loss_val,
                                      logger, config, 'train')

        if is_print:
            writer.add_scalars("Loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch + 1)

        # lr_scheduler.step_update(epoch)

    torch.distributed.destroy_process_group()


def reduce_tensor(tensor):
    # https://github.com/NVIDIA/DeepLearningExamples/blob/777d174008c365a5d62799a86f135a4f171f620e/PyTorch/Classification/ConvNets/image_classification/utils.py#L117-L123
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # SUM
    # rt /= (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    return rt


def train(model, device, train_dataloader, scaler, optimizer, epoch, is_print, is_obs, max_norm):
    model.train()
    loss_train = 0

    train_dataloader.sampler.set_epoch(epoch)

    for batch_idx, data_i in enumerate(train_dataloader):

        optimizer.zero_grad(set_to_none=True)

        # Runs the forward pass with autocasting
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(data_i['hres_forecast'].to(device),
                            data_i['glofas'].to(device),
                            data_i['era5'].to(device),
                            data_i['cpc'].to(device),
                            data_i['static'].to(device),
                            data_i['curves'].to(device).long(),
                            data_i['glofas_target'].to(device) if not is_obs else None,
                            data_i['obs_target'].to(device) if is_obs else None,
                            data_i['weight'].to(device),
                            )

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = torch.nan_to_num(loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        loss_train += loss.item()

        if is_print and batch_idx % 10 == 0:  # and device == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_dataloader),
                100. * batch_idx / len(train_dataloader), loss.item()))

    mean_loss_train = loss_train / float(len(train_dataloader))

    return mean_loss_train, loss_train


def val(model, device, val_dataloader, epoch, is_print, is_obs):

    model.eval()
    loss_val = 0

    # validation
    with torch.no_grad():

        val_dataloader.sampler.set_epoch(epoch)

        for batch_idx, data_i in enumerate(val_dataloader):

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(data_i['hres_forecast'].to(device),
                                data_i['glofas'].to(device),
                                data_i['era5'].to(device),
                                data_i['cpc'].to(device),
                                data_i['static'].to(device),
                                data_i['curves'].to(device).long(),
                                data_i['glofas_target'].to(device) if not is_obs else None,
                                data_i['obs_target'].to(device) if is_obs else None,
                                data_i['weight'].to(device),
                                )

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = torch.nan_to_num(loss)

            loss_val += loss.item()

            if is_print and batch_idx % 10 == 0:  # and device == 0:
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(val_dataloader),
                    100. * batch_idx / len(val_dataloader), loss.item()))

        mean_loss_val = loss_val / float(len(val_dataloader))

    return mean_loss_val, loss_val


if __name__ == "__main__":
    main()
