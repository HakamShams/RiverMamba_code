# ------------------------------------------------------------------
# Script for training and validating on GloFAS using LSTM
# 
# Description:
#   - This script trains a hydrological forecasting model on GloFAS streamflow data
#     using a bidirectional hindcast-forecast LSTM blocks with weighted loss function.
#   - It applies on pretraining on era5, using era5 reanalysis data and GloFAS reanalysis data.
# 
# Features:
#   - Data sources: GloFAS, ERA5-Land, HRES forecasts (optional), CPC, static variables.
#   - Dataset class: GloFAS_Dataset_edlstm supports flexible input combinations and options.
#   - Model: Google-style encoder-decoder LSTM with optional state handoff and weighted loss.
#   - Mixed precision training via torch.cuda.amp.
#   - Early stopping and model checkpointing based on loss, F1-score, and R².
#   - Multi-GPU training with DataParallel support.
#   - TensorBoard logging and detailed GPU memory usage tracing.
#
#
#  ------------------------------------------------------------------
import torch
import numpy as np
from tqdm import tqdm
import RiverMamba.lstm_repo.utils.utils_edlstm as utils
from RiverMamba.lstm_repo.models_lstm.build_lstm import Model
import time
import os
from torch.utils.tensorboard import SummaryWriter
# this is for hres-based forecast
from RiverMamba.lstm_repo.dataset.LSTM_dataset import GloFAS_Dataset_edlstm

import RiverMamba.lstm_repo.config_lstm.config_edlstm_era5 as config_file
from torch.cuda.amp import GradScaler
from torch import autocast
from typing import Dict, List, Optional

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

import subprocess

def iter_time_gen(dataloader):
    dataloader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        t0 = time.time()
        batch = next(dataloader_iter)
        yield batch, t0


def get_gpu_memory_map():
    """Get the current GPU memory usage."""
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
        encoding='utf-8'
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def log_gpu_memory(stage):
    gpu_memory = get_gpu_memory_map()
    print(f"{stage} - GPU memory usage: {gpu_memory} MB")

# torch.autograd.set_detect_anomaly(True)
# ------------------------------------------------------------------

def train(config_file):
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print (f"num of gpus: {torch.cuda.device_count()}")
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
    train_dataset = GloFAS_Dataset_edlstm(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_static=config.root_static,
        root_obs    = config.root_obs,
        root_cpc   = config.root_cpc,
        is_hres_forecast = config.is_hres_forecast, # by default False
        nan_fill=config.nan_fill,
        delta_t=config.delta_t, # the length of the input -- it generates a continueous time series
        delta_t_f=config.delta_t_f, # forecast length
        is_random_t=config.is_random_t, #False,
        is_aug=config.is_aug, #False,
        is_shuffle=config.is_shuffle, #False,
        variables_glofas=config.variables_glofas,
        variables_era5_land=config.variables_era5_land,
        variables_static=config.variables_static,
        variables_hres_forecast=config.variables_hres_forecast,
        variables_glofas_log1p=config.variables_glofas_log1p, # None
        variables_era5_land_log1p=config.variables_era5_land_log1p, # None
        variables_static_log1p=config.variables_static_log1p,
        variables_hres_forecast_log1p=config.variables_hres_forecast_log1p,
        is_norm=config.is_norm, # True/False
        years=config.years_train, # years_train
        lat_min=config.lat_min, # 30
        lat_max=config.lat_max, # 60
        lon_min=config.lon_min, # -10
        lon_max=config.lon_max, # 40
        is_target_obs=config.is_target_obs # here should be False
    )

    print('number of sampled data:', train_dataset.__len__()) # number of files in the dataset

    utils.log_string(logger, "loading validation dataset ...")
    val_dataset = GloFAS_Dataset_edlstm(
        root_glofas_reanalysis=config.root_glofas_reanalysis,
        root_era5_land_reanalysis=config.root_era5_land_reanalysis,
        root_hres_forecast=config.root_hres_forecast,
        root_static=config.root_static,
        root_obs=config.root_obs,
        root_cpc=config.root_cpc,
        is_hres_forecast=config.is_hres_forecast,
        nan_fill=config.nan_fill,
        delta_t=config.delta_t, # define the length of the input -- it generates a continuous time series
        delta_t_f=config.delta_t_f, # define forecast length
        is_random_t=config.is_random_t, #False
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
        years=config.years_val, # years_val
        lat_min=config.lat_min, # 30
        lat_max=config.lat_max, # 60
        lon_min=config.lon_min, # -10
        lon_max=config.lon_max, # 40
        is_target_obs=config.is_target_obs
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size_train,
                                                   #drop_last=True, # handling the smaller last batch
                                                   shuffle=True,
                                                   pin_memory=config.pin_memory, # what is this?
                                                   num_workers=config.n_workers_train) # what is this?

    print("Train DataLoader initialized.")


    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config.batch_size_val,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=config.pin_memory,
                                                 num_workers=config.n_workers_val)

    utils.log_string(logger, "# training samples: %d" % len(train_dataset))
    utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    # Define the device

    if config.gpu_id != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = 'cuda'
        use_amp = True  # Use Automatic Mixed Precision
    else:
        device = 'cpu'
        use_amp = False  # Disable AMP for CPU

    # input should come from the dataloader
    if config.is_hres_forecast: # if set to fine-tune, use the is_hres_forecast instead of era5_land
        model = Model(
            data_glofas_size_h=len(config.variables_glofas),
            data_era5_size_h=len(config.variables_era5_land), # the initialization part should avoid using the data
            data_size_f=len(config.variables_hres_forecast), # needs to be changed -- the name not precise
            data_cpc_size=1, # add the cpc data for input
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
    
    else:
        model = Model(
            data_glofas_size_h=len(config.variables_glofas),
            data_era5_size_h=len(config.variables_era5_land), # the initialization part should avoid using the data
            data_size_f=len(config.variables_hres_forecast), # needs to be changed
            data_cpc_size=1, # add the cpc data for input
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

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # get optimizer
    utils.log_string(logger, "get optimizer and learning rate scheduler ...")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay,
                                 betas=(config.beta1, config.beta2))
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip the gradient

    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    model.to(device) # first move the model to the device

    # one node multiple GPUs Parallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model,device_ids=[0,1,2,3])

    # === Resume checkpoint if exists ===
    start_epoch = 0
    resume_path = os.path.join(config.dir_model, 'latest.pth')

    if getattr(config, 'resume', False):
        if os.path.exists(resume_path):
            print(f"[RESUME] Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)

            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[RESUME] Resumed training from epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"[ERROR] Resume checkpoint not found at: {resume_path}")
    else:
        print("[INIT] Starting training from scratch.")


    scaler = GradScaler() if use_amp else None  # Only use GradScaler when AMP is enabled
    # Log GPU memory usage after model initialization
    log_gpu_memory("After model initialization")
    # training loop
    utils.log_string(logger, 'training on GloFAS dataset ...\n')

    eval_train = utils.evaluator(logger, 'Training', lead_time=train_dataset.delta_t_f)
    eval_val = utils.evaluator(logger, 'Validation', lead_time=train_dataset.delta_t_f)

    # initialize the best values
    best_loss_train = np.inf
    best_loss_val = np.inf
    best_F1_val = 0
    best_R2_val = 0

    lead_time = config.delta_t_f

    # Early Stopping Parameters
    patience = config.patience
    counter = 0     # Counter to track epochs without improvement
    
    for epoch in range(start_epoch, config.n_epochs):

        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config.n_epochs))
        model.train()
        loss_train = 0
        start_epoch_time = time.time()
        # Manage the data shape
        # deal with the dataset
        # Log GPU memory usage at the start of the epoch
        log_gpu_memory(f"Start of training epoch {epoch}")

        train_dataloader_iter = iter(train_dataloader)
        for i in tqdm(range(len(train_dataloader)), total=len(train_dataloader), smoothing=0.9, postfix="training"):
            start_time = time.time()
            data_glofas, data_era5_land, data_forecast, data_static, data_cpc, target_obs, \
            data_thresholds, target_glofas, _, _ = next(train_dataloader_iter)
            # Log GPU memory usage after loading data
             
            if (i + 1) % 500 == 0:
                log_gpu_memory("After loading data")
                print (f"batch count: {i+1}")
            # input from dataloader is (batch,point,seq, feature)

            # reshape the dataset by combining the batch and sampling points -- batch*points
            batch_size_actual = data_glofas.shape[0]  # Use actual batch size -- avoid the mismatch of shape from last batch
            data_glofas = data_glofas.view(batch_size_actual*config.n_points,data_glofas.shape[2],-1) # (batch*points, seq, features) 
            data_era5_land = data_era5_land.view(batch_size_actual*config.n_points,data_era5_land.shape[2],-1)
            data_forecast = data_forecast.view(batch_size_actual*config.n_points,data_forecast.shape[2],-1)
            data_cpc = data_cpc.view(batch_size_actual*config.n_points,data_cpc.shape[2],-1)
            
            data_static = data_static.view(batch_size_actual*config.n_points,-1)
            
            data_thresholds=data_thresholds.view(batch_size_actual*config.n_points,-1)

            if config.is_target_obs:

                target = target_obs.view(batch_size_actual*config.n_points,target_obs.shape[2],-1)

            else:

                target = target_glofas.view(batch_size_actual*config.n_points,target_glofas.shape[2],-1)

            # Convert all tensors to float32 -- dataloader
            data_glofas = data_glofas.to(torch.float32)
            data_era5_land = data_era5_land.to(torch.float32)
            data_forecast = data_forecast.to(torch.float32)
            data_static = data_static.to(torch.float32)  
            data_cpc = data_cpc.to(torch.float32)
            target = target.to(torch.float32)
            data_thresholds = data_thresholds.to(torch.float32)

            # Runs the forward pass with autocasting.
            if use_amp:
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred, loss = model(data_glofas.to(device),
                                    data_era5_land.to(device),
                                    data_cpc.to(device),
                                    data_forecast.to(device),
                                    data_static.to(device),
                                    target.to(device),is_obs=config.is_target_obs)
                    loss = loss.mean()
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()

                # Apply gradient clipping (if needed)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iterat ion.
                scaler.update()

                # Zero gradients after optimizer step
                optimizer.zero_grad(set_to_none=True)
            else:
                pred, loss = model(data_glofas.to(device),
                                    data_era5_land.to(device),
                                    data_cpc.to(device),
                                    data_forecast.to(device),
                                    data_static.to(device),
                                    target.to(device),is_obs=config.is_target_obs)

                loss = loss.mean()
                loss.backward()

                # Compute and log gradient norm
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

                # Apply gradient clipping (if needed)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # **Step the optimizer to update the model parameters**
                optimizer.step()

                # shape of pred and target is (batch,forecast_seq, var)

            loss_train += loss.item()

            # Print timing for this batch iteration
            batch_duration = time.time() - start_time
            print(f"Batch {i + 1} Training Time: {batch_duration:.4f} seconds")

            target = target.numpy().astype(np.float64) # (batch, seq_length, 1)
            pred = pred.detach().cpu().numpy().astype(np.float64) # (batch, seq_length, 1)

            if config.is_norm:
                
                data_glofas = train_dataset.inv_transform(data_glofas, train_dataset.glofas_mean, train_dataset.glofas_std)
            
            start = time.time()
            pred = train_dataset.log1p_inv_transform(pred)
            target = train_dataset.log1p_inv_transform(target)
            if (i + 1) % 500 == 0:
                print(f"inv_transform total time: {time.time() - start:.4f}s")
           
            # Since reset() is not called after processing each batch, 
            # the metrics keep accumulating across batches during the epoch 
            if isinstance(data_glofas, np.ndarray):
                data_glofas = torch.tensor(data_glofas, dtype=torch.float32, device='cpu')

            # Convert to NumPy only after selecting the desired slice
            data_glofas = data_glofas[:, -1, 1:1+1].detach().numpy()
            
            # Log GPU memory usage after processing a batch
            if (i + 1) % 500 == 0:
                log_gpu_memory("After processing a batch for training")

        post_train_time = time.time() 

        mean_loss_train = loss_train / float(len(train_dataloader))

        utils.log_string(logger, '%s mean loss     : %.6f' % ('Training', mean_loss_train))
        utils.log_string(logger, '%s best mean loss: %.6f\n' % ('Training', best_loss_train))

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train
        
        print(f"post training epoch Time: {batch_duration:.4f} seconds")

        # validation
        print ("validation begins:")
        with torch.no_grad():

            model.eval()
            loss_val = 0

            val_dataloader_iter = iter(val_dataloader)
            for i in tqdm(range(len(val_dataloader)), total=len(val_dataloader), smoothing=0.9, postfix="training"):
                start_time = time.time()
                data_glofas, data_era5_land, data_forecast, data_static, data_cpc, target_obs, \
                data_thresholds, target_glofas, _, _ = next(val_dataloader_iter)
                
                
                optimizer.zero_grad(set_to_none=True)
                
                # reshape the dataset by combining dimension batch and points
                
                batch_size_actual = data_glofas.shape[0]  # Use actual batch size

                data_glofas = data_glofas.view(batch_size_actual*config.n_points,data_glofas.shape[2],-1) # (batch*points, seq, features) 
                data_era5_land = data_era5_land.view(batch_size_actual*config.n_points,data_era5_land.shape[2],-1)
                data_forecast = data_forecast.view(batch_size_actual*config.n_points,data_forecast.shape[2],-1)
                data_cpc = data_cpc.view(batch_size_actual*config.n_points,data_cpc.shape[2],-1)
                data_static = data_static.view(batch_size_actual*config.n_points,-1)
                data_thresholds=data_thresholds.view(batch_size_actual*config.n_points,-1)

                if config.is_target_obs:

                    target = target_obs.view(batch_size_actual*config.n_points,target_obs.shape[2],-1)

                else:

                    target = target_glofas.view(batch_size_actual*config.n_points,target_glofas.shape[2],-1)
                
                # Convert all tensors to float32
                data_glofas = data_glofas.to(torch.float32)
                data_era5_land = data_era5_land.to(torch.float32)
                data_forecast = data_forecast.to(torch.float32)
                data_static = data_static.to(torch.float32)  
                data_cpc = data_cpc.to(torch.float32)
                data_thresholds = data_thresholds.to(torch.float32)
                target = target.to(torch.float32)

                if use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        pred, loss = model(data_glofas.to(device),
                                    data_era5_land.to(device),
                                    data_cpc.to(device),
                                    data_forecast.to(device),
                                    data_static.to(device),
                                    target.to(device),is_obs=config.is_target_obs)
                        loss = loss.mean()

                else:
                    
                    pred, loss = model(data_glofas.to(device),
                                    data_era5_land.to(device),
                                    data_cpc.to(device),
                                    data_forecast.to(device),
                                    data_static.to(device),
                                    target.to(device),is_obs=config.is_target_obs)
                                    
                    loss = loss.mean()

                loss_val += loss.item()

                batch_time = time.time() - start_time
                if (i + 1) % 500 == 0:
                    print(f"Batch val {i + 1} iteration time: {batch_time} seconds")

                target = target.numpy().astype(np.float64)
                pred = pred.cpu().numpy().astype(np.float64)

                if config.is_norm:

                    data_glofas = train_dataset.inv_transform(data_glofas, train_dataset.glofas_mean, train_dataset.glofas_std)
                
                pred = train_dataset.log1p_inv_transform(pred)
                target = train_dataset.log1p_inv_transform(target)

                if isinstance(data_glofas, np.ndarray):
                    data_glofas = torch.tensor(data_glofas, dtype=torch.float32, device='cpu')

                # Convert to NumPy only after selecting the desired slice
                data_glofas = data_glofas[:, -1, 1:1+1].detach().numpy()

                # Measure the time taken for validation
                start_time = time.time()

                eval_val(pred,
                         target,  data_glofas,
                         data_thresholds.numpy())
                
                validation_time = time.time() - start_time
                if (i + 1) % 500 == 0:
                    print(f"val eval time: {validation_time} seconds")


        mean_loss_val = loss_val / float(len(val_dataloader))
        eval_val.get_results()

        utils.log_string(logger, '%s mean loss     : %.6f' % ('Validation', mean_loss_val))
        utils.log_string(logger, '%s best mean loss: %.6f\n' % ('Validation', best_loss_val))

        # ========================= Early stopping & model saving ========================= # 
        min_delta = 1e-4  # set a tolerance for minimal improvement

        if mean_loss_val < best_loss_val - min_delta:
            best_loss_val = mean_loss_val
            counter = 0  # reset early stopping counter
            utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'loss')
        else:
            counter += 1
            print(f"Early Stopping Counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered!")
                break

        # Still save best F1 and R2 models independently
        if eval_val.F1 >= best_F1_val:
            best_F1_val = eval_val.F1
            utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'F1')

        if eval_val.dis_r2 >= best_R2_val:
            best_R2_val = eval_val.dis_r2
            utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'R2')

        writer.add_scalars("Loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch)
        print(f"Epoch {epoch}: Train Loss = {mean_loss_train}, Val Loss = {mean_loss_val}")

        eval_train.reset()
        eval_val.reset()

        lr_scheduler.step_update(epoch)

        epoch_duration = time.time() - start_epoch_time
        log_gpu_memory(f"End of epoch {epoch+1}")
        print(f"Epoch {epoch + 1} duration: {epoch_duration} seconds")

        # Always save checkpoint -- as latest
        utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'latest')
        
    # Log GPU memory usage after training
    log_gpu_memory("After training")


if __name__ == '__main__':
    train(config_file)