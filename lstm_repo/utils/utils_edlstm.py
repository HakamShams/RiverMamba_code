# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import numpy as np
import random
import os
import datetime
import logging
from functools import reduce
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import r2_score#, mean_squared_error, mean_absolute_error

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


# ------------------------------------------------------------------
def log_string(logger, str):
    logger.info(str)
    print(str)

def safe_metric(metric_fn, y_pred, y_true):
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if np.sum(mask) == 0:
        return np.nan
    return metric_fn(y_pred[mask], y_true[mask])

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_logger(config):
    # Set Logger and create Directories

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    make_dir(dir_log)

    if config.phase == 'train':
        checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
        make_dir(checkpoints_dir)

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % dir_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer_groups(model, config):
    # Based on https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()

    blacklist_weight_modules = (torch.nn.LayerNorm,
                                torch.nn.Embedding,
                                torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d,
                                torch.nn.BatchNorm1d,
                                torch.nn.GroupNorm,
                                torch.nn.InstanceNorm1d,
                                torch.nn.InstanceNorm2d,
                                torch.nn.InstanceNorm3d)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name##
            mm = reduce(getattr, fpn.split(sep='.')[:-1], model)
            if pn.endswith('bias') or pn.endswith('relative_position_bias_table'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(mm, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [

        {"params": [param_dict[pn] for pn in sorted(list(decay))],
         'lr': config.en_lr, "weight_decay": config.en_weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
         'lr': config.en_lr, "weight_decay": 0.0},

    ]

    return optim_groups


def get_optimizer(optim_groups, config):
    optim = config.optimizer

    if optim == 'Adam':
        optimizer = torch.optim.Adam(optim_groups, betas=(config.beta1, config.beta2))
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups, betas=(config.beta1, config.beta2))
  
    else:
        raise ValueError('Unexpected optimizer {} supported optimizers are Adam and AdamW'.format(config.optimizer))

    return optimizer


def get_learning_scheduler(optimizer, config):
    lr_scheduler = config.lr_scheduler

    if lr_scheduler == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config.lr_decay_step,
            decay_rate=config.lr_decay_rate,
            warmup_lr_init=config.lr_warmup,
            warmup_t=config.lr_warmup_epochs,
            t_in_epochs=True,
        )

    elif lr_scheduler == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config.n_epochs,
            cycle_mul=1.,
            lr_min=config.lr_min,
            warmup_lr_init=config.lr_warmup,
            warmup_t=config.lr_warmup_epochs,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=False
        )

    else:
        raise ValueError('Unexpected optimizer {}, supported scheduler is step or cosine'.format(config.optimizer))

    return lr_scheduler


def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculated the r-squared score between 2 arrays of values

    Args:
        y_pred (np.ndarray): Predicted values
        y_true (np.ndarray): Target values

    Returns:
         (float) r-squared score
    """
    index = ~np.logical_or(np.isnan(y_pred), np.isinf(y_pred))

    return r2_score(y_true[index], y_pred[index])


# ...existing code...

def bias_ratio(y_pred, y_true):
    """
    Compute the bias ratio (beta) defined as the ratio of the mean of predictions
    to the mean of observations.

    beta = mean(y_pred) / mean(y_true)

    Parameters:
    -----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        True observed values.

    Returns:
    --------
    beta : float
        The bias ratio. Returns np.nan if the mean of y_true is zero.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    
    if true_mean == 0:
        return np.nan
    
    beta = pred_mean / true_mean
    return beta

def pearson_r(y_pred, y_true):
    """
    Compute the Pearson correlation coefficient between predictions and observations.

    Parameters:
    -----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        True observed values.

    Returns:
    --------
    r : float
        Pearson correlation coefficient.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Calculate means
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    
    # Calculate covariance numerator and denominator
    numerator = np.sum((y_pred - pred_mean) * (y_true - true_mean))
    denominator = np.sqrt(np.sum((y_pred - pred_mean)**2) * np.sum((y_true - true_mean)**2))
    
    # Avoid division by zero
    if denominator == 0:
        return np.nan
    
    r = numerator / denominator
    return r

def variability_ratio(y_pred, y_true):
    """
    Compute the variability ratio (γ) defined as the ratio of the coefficient of variation
    of predictions to that of observations.

    γ = (std(y_pred)/mean(y_pred)) / (std(y_true)/mean(y_true))

    Parameters:
    -----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        True observed values.

    Returns:
    --------
    gamma : float
        Variability ratio.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Calculate means and standard deviations
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    
    # Check for division by zero in means
    if pred_mean == 0 or true_mean == 0:
        return np.nan
    
    # Calculate coefficients of variation
    cv_pred = pred_std / pred_mean
    cv_true = true_std / true_mean
    
    # Avoid division by zero for cv_true
    if cv_true == 0:
        return np.nan
    
    gamma = cv_pred / cv_true
    return gamma

# ...existing code...

def kge_multi(y_pred, y_true):
    """
    Modified Kling-Gupta Efficiency (KGE) and its three components (r, γ, β) as per Kling et al., 2012
    https://doi.org/10.1016/j.jhydrol.2012.01.011
    """
    # check for nan and inf values
    index = ~np.logical_or(np.isnan(y_pred), np.isinf(y_pred))
    y_true = y_true[index]
    y_pred = y_pred[index]

    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    pred_mean = np.mean(y_pred, dtype=np.float64)
    target_mean = np.mean(y_true, dtype=np.float64)

    r_num = np.sum((y_pred - pred_mean) * (y_true - target_mean), dtype=np.float64)
    r_den = np.sqrt(np.sum((y_pred - pred_mean) ** 2, dtype=np.float64)
                    * np.sum((y_true - target_mean) ** 2, dtype=np.float64))
    r = r_num / r_den

    # calculate error in volume beta (bias of mean discharge)
    beta = (np.mean(y_pred, dtype=np.float64)
            / np.mean(y_true, dtype=np.float64))

    # calculate error in spread of flow gamma
    gamma = ((np.std(y_pred, dtype=np.float64) / pred_mean)
             / (np.std(y_true, dtype=np.float64) / target_mean))

    # calculate the modified Kling-Gupta Efficiency KGE'
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    return kge


class evaluator_alt():
    def __init__(self, logger, mode, lead_time: int = 1):

        self.classes = ['return period - ' + t for t in ["1.5", "2  ", "5  ", "10 ", "20 ", "50 ", "100", "200", "500"]]
        self.n_classes = len(self.classes)

        self.lead_time = lead_time
        self.lead_time_classes_t = [str(t+1) if t >= 9 else "0" + str(t+1) for t in range(lead_time)]
        self.lead_time_classes = [u'Δt_' + str(t * 24) for t in range(lead_time)]

        self.mode = mode
        self.logger = logger

        self.seen_iter = 0
        self.dis_mae, self.dis_rmse, self.dis_r2, self.dis_kge = 0, 0, 0, 0

        self.dis_lead_time_mae = [0 for _ in range(lead_time)]
        self.dis_lead_time_rmse = [0 for _ in range(lead_time)]
        self.dis_lead_time_r2 = [0 for _ in range(lead_time)]
        self.dis_lead_time_kge = [0 for _ in range(self.lead_time)]

        self.seen_all = 0
        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

    def get_results(self):

        for t in range(self.lead_time):
            self.dis_lead_time_mae[t] = self.dis_lead_time_mae[t] / float(self.seen_iter)
            self.dis_lead_time_rmse[t] = self.dis_lead_time_rmse[t] / float(self.seen_iter)
            self.dis_lead_time_r2[t] = self.dis_lead_time_r2[t] / float(self.seen_iter)
            self.dis_lead_time_kge[t] = self.dis_lead_time_kge[t] / float(self.seen_iter)

        weights_label = self.weights_label.astype(np.float32) / float(self.seen_all)

        accuracy = [0 for _ in range(self.n_classes)]
        precision = [0 for _ in range(self.n_classes)]
        F1 = [0 for _ in range(self.n_classes)]
        iou = [0 for _ in range(self.n_classes)]

        for label in range(self.n_classes):
            precision[label] = self.correct_label_all[label] / float(self.predicted_label_all[label])
            accuracy[label] = self.correct_label_all[label] / (np.array(self.seen_label_all[label], dtype=float) + 1e-6)
            F1[label] = 2 * precision[label] * accuracy[label] / (accuracy[label] + precision[label])
            iou[label] = self.correct_label_all[label] / float(self.iou_de_label_all[label])

        message = '-----------------   %s   -----------------\n' % self.mode

        for t in range(self.lead_time):
            message += 'lead time %s - %s   MAE: %.6f, RMSE: %.6f, R2: %.4f, KGE: %.4f \n' % (
                self.lead_time_classes_t[t],
                self.lead_time_classes[t] + ' ' * (7 - len(self.lead_time_classes[t])),
                self.dis_lead_time_mae[t],
                self.dis_lead_time_rmse[t],
                self.dis_lead_time_r2[t],
                self.dis_lead_time_kge[t]
            )

        message += '\n'

        if self.lead_time > 1:

            self.dis_mae = self.dis_mae / float(self.seen_iter)
            self.dis_rmse = self.dis_rmse / float(self.seen_iter)
            self.dis_r2 = self.dis_r2 / float(self.seen_iter)

            message += 'lead time overall        MAE: %.6f, RMSE: %.6f, R2: %.4f, KGE: %.4f \n' % (
                self.dis_mae,
                self.dis_rmse,
                self.dis_r2,
                self.dis_kge)

        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' ' * (14 - len(self.classes[label])), weights_label[label],
                precision[label],
                accuracy[label],
                F1[label],
                iou[label])

        log_string(self.logger, message)

    def reset(self):

        self.seen_iter = 0
        self.dis_mae, self.dis_rmse, self.dis_r2, self.dis_kge = 0, 0, 0, 0

        self.dis_lead_time_mae = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_rmse = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_r2 = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_kge = [0 for _ in range(self.lead_time)]

        self.seen_all = 0
        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

    def __call__(self, pred, target, thresholds):

        # pred B, P, C
        # target B, P, C
        # thresholds B, P, 9

        if self.lead_time > 1:
            self.dis_mae += np.mean(np.abs(pred - target))
            self.dis_rmse += np.sqrt(np.mean((pred - target) ** 2))
            self.dis_r2 += r2_score_multi(pred.flatten(), target.flatten())
            self.dis_kge += kge_multi(pred.flatten(), target.flatten())

        for t in range(self.lead_time):
            self.dis_lead_time_mae[t] += np.mean(np.abs(pred[:, :, t] - target[:, :, t]))
            self.dis_lead_time_rmse[t] += np.sqrt(np.mean((pred[:, :, t] - target[:, :, t]) ** 2))
            self.dis_lead_time_r2[t] += r2_score_multi(pred[:, :, t].flatten(), target[:, :, t].flatten())
            self.dis_lead_time_kge[t] += kge_multi(pred[:, :, t].flatten(), target[:, :, t].flatten())

        pred_thr = (np.repeat(pred[:, :, 0:1], 9, axis=-1) >= thresholds)
        target_thr = (np.repeat(target[:, :, 0:1], 9, axis=-1) >= thresholds)

        self.seen_all += np.prod(target_thr[:, :, 0].shape)
        self.weights_label += np.sum(target_thr, axis=(0, 1))

        for label in range(self.n_classes):
            self.correct_label_all[label] += np.sum((pred_thr[:, :, label] == 1) & (target_thr[:, :, label] == 1))
            self.seen_label_all[label] += np.sum((target_thr[:, :, label] == 1))
            self.iou_de_label_all[label] += np.sum(((pred_thr[:, :, label] == 1) | (target_thr[:, :, label] == 1)))
            self.predicted_label_all[label] += np.sum(pred_thr[:, :, label] == 1)

        self.seen_iter += 1


class evaluator():
    def __init__(self, logger, mode, lead_time: int = 7):

        self.classes = ['return period - ' + t for t in ["1.5", "2  ", "5  ", "10 ", "20 ", "50 ", "100", "200", "500"]]
        self.n_classes = len(self.classes)

        self.lead_time = lead_time
        self.lead_time_classes_t = [str(t+1) if t >= 9 else "0" + str(t+1) for t in range(lead_time)]
        self.lead_time_classes = [u'Δt_' + str((t + 1) * 24) for t in range(lead_time)]

        self.mode = mode
        self.logger = logger

        self.seen_iter = 0
        self.dis_mae, self.dis_rmse, self.dis_r2, self.dis_kge = 0, 0, 0, 0
        self.dis_pearson_r, self.dis_variability_ratio, self.dis_bias_ratio = 0, 0, 0

        self.seen_lead_time_iter = [0 for _ in range(lead_time)]
        self.dis_lead_time_mae = [0 for _ in range(lead_time)]
        self.dis_lead_time_rmse = [0 for _ in range(lead_time)]
        self.dis_lead_time_r2 = [0 for _ in range(lead_time)]
        self.dis_lead_time_kge = [0 for _ in range(lead_time)]
        self.dis_lead_time_pearson_r = [0 for _ in range(lead_time)]
        self.dis_lead_time_variability_ratio = [0 for _ in range(lead_time)]
        self.dis_lead_time_bias_ratio = [0 for _ in range(lead_time)]

        self.seen_all = 0
        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

        self.seen_lead_time_all = [0 for _ in range(lead_time)]
        self.weights_lead_time_label = np.zeros((lead_time, self.n_classes))
        self.seen_lead_time_label_all = np.zeros((lead_time, self.n_classes))
        self.correct_lead_time_label_all = np.zeros((lead_time, self.n_classes))
        self.iou_de_lead_time_label_all = np.zeros((lead_time, self.n_classes))
        self.predicted_lead_time_label_all = np.zeros((lead_time, self.n_classes))


    def get_results(self):

        weights_label = self.weights_label.astype(np.float32) / float(self.seen_all)

        accuracy = [0 for _ in range(self.n_classes)]
        precision = [0 for _ in range(self.n_classes)]
        F1 = [0 for _ in range(self.n_classes)]
        iou = [0 for _ in range(self.n_classes)]

        for label in range(self.n_classes):
            precision[label] = self.correct_label_all[label] / float(self.predicted_label_all[label])
            accuracy[label] = self.correct_label_all[label] / (np.array(self.seen_label_all[label], dtype=float) + 1e-6)
            F1[label] = 2 * precision[label] * accuracy[label] / (accuracy[label] + precision[label])
            iou[label] = self.correct_label_all[label] / float(self.iou_de_label_all[label])

        accuracy_lead_time = np.zeros((self.lead_time, self.n_classes))
        precision_lead_time = np.zeros((self.lead_time, self.n_classes))
        F1_lead_time = np.zeros((self.lead_time, self.n_classes))
        iou_lead_time = np.zeros((self.lead_time, self.n_classes))

        weights_lead_time_label = np.zeros((self.lead_time, self.n_classes))

        for t in range(self.lead_time):
            if self.seen_lead_time_iter[t] == 0:
                self.seen_lead_time_iter[t] = 1

            weights_lead_time_label[t] = (self.weights_lead_time_label.astype(np.float32)[t] /
                                          float(self.seen_lead_time_all[t]))

            self.dis_lead_time_mae[t] = self.dis_lead_time_mae[t] / float(self.seen_lead_time_iter[t])
            self.dis_lead_time_rmse[t] = self.dis_lead_time_rmse[t] / float(self.seen_lead_time_iter[t])
            self.dis_lead_time_r2[t] = self.dis_lead_time_r2[t] / float(self.seen_lead_time_iter[t])
            self.dis_lead_time_kge[t] = self.dis_lead_time_kge[t] / float(self.seen_lead_time_iter[t])
            self.dis_lead_time_pearson_r[t] = self.dis_lead_time_pearson_r[t] / float(self.seen_lead_time_iter[t])
            self.dis_lead_time_variability_ratio[t] = self.dis_lead_time_variability_ratio[t] / float(self.seen_lead_time_iter[t])
            self.dis_lead_time_bias_ratio[t] = self.dis_lead_time_bias_ratio[t] / float(self.seen_lead_time_iter[t])

            for label in range(self.n_classes):
                precision_lead_time[t, label] = self.correct_lead_time_label_all[t, label] / float(self.predicted_lead_time_label_all[t, label])
                accuracy_lead_time[t, label] = self.correct_lead_time_label_all[t, label] / (
                            np.array(self.seen_lead_time_label_all[t, label], dtype=float) + 1e-6)
                F1_lead_time[t, label] = (2 * precision_lead_time[t, label] * accuracy_lead_time[t, label] /
                                          (accuracy_lead_time[t, label] + precision_lead_time[t, label]))
                iou_lead_time[t, label] = (self.correct_lead_time_label_all[t, label] /
                                           float(self.iou_de_lead_time_label_all[t, label]))

        message = '-----------------   %s   -----------------\n' % self.mode

        for t in range(self.lead_time):
            message += 'lead time %s - %s   MAE: %.6f, RMSE: %.6f, R2: %.4f, KGE: %.4f, Pearson R: %.4f, Variability Ratio: %.4f, Bias Ratio: %.4f \n' % (
                self.lead_time_classes_t[t],
                self.lead_time_classes[t] + ' ' * (7 - len(self.lead_time_classes[t])),
                self.dis_lead_time_mae[t],
                self.dis_lead_time_rmse[t],
                self.dis_lead_time_r2[t],
                self.dis_lead_time_kge[t],
                self.dis_lead_time_pearson_r[t],
                self.dis_lead_time_variability_ratio[t],
                self.dis_lead_time_bias_ratio[t]
            )

        message += '\n'

        self.dis_mae = self.dis_mae / float(self.seen_iter) # normalize the total number of batch
        self.dis_rmse = self.dis_rmse / float(self.seen_iter)
        self.dis_r2 = self.dis_r2 / float(self.seen_iter)
        self.dis_kge = self.dis_kge / float(self.seen_iter)
        self.dis_pearson_r = self.dis_pearson_r / float(self.seen_iter)
        self.dis_variability_ratio = self.dis_variability_ratio / float(self.seen_iter)
        self.dis_bias_ratio = self.dis_bias_ratio / float(self.seen_iter)

        message += 'lead time overall        MAE: %.6f, RMSE: %.6f, R2: %.4f, KGE: %.4f, Pearson R: %.4f, Variability Ratio: %.4f, Bias Ratio: %.4f \n' % (
            self.dis_mae,
            self.dis_rmse,
            self.dis_r2,
            self.dis_kge,
            self.dis_pearson_r,
            self.dis_variability_ratio,
            self.dis_bias_ratio)

        message += '\n'

        for t in range(self.lead_time):
            message += ('lead time %s - %s   F1 1.5: %.4f, F1 2: %.4f, F1 5: %.4f, F1 10: %.4f, F1 20: %.4f, '
                        'F1 50: %.4f, F1 100: %.4f, F1 200: %.4f, F1 500: %.4f \n') % (
                self.lead_time_classes_t[t],
                self.lead_time_classes[t] + ' ' * (7 - len(self.lead_time_classes[t])),
                F1_lead_time[t, 0],
                F1_lead_time[t, 1],
                F1_lead_time[t, 2],
                F1_lead_time[t, 3],
                F1_lead_time[t, 4],
                F1_lead_time[t, 5],
                F1_lead_time[t, 6],
                F1_lead_time[t, 7],
                F1_lead_time[t, 8],
            )

        message += '\n'
        message += 'Overall classification scores across lead times\n\n'

        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' ' * (14 - len(self.classes[label])), weights_label[label],
                precision[label],
                accuracy[label],
                F1[label],
                iou[label])

        self.F1 = np.nanmean(F1)

        log_string(self.logger, message)

    def reset(self):

        self.seen_iter = 0
        self.dis_mae, self.dis_rmse, self.dis_r2, self.dis_kge = 0, 0, 0, 0
        self.dis_pearson_r, self.dis_variability_ratio, self.dis_bias_ratio = 0, 0, 0

        self.seen_lead_time_iter = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_mae = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_rmse = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_r2 = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_kge = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_pearson_r = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_variability_ratio = [0 for _ in range(self.lead_time)]
        self.dis_lead_time_bias_ratio = [0 for _ in range(self.lead_time)]

        self.seen_all = 0
        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

        self.seen_lead_time_all = [0 for _ in range(self.lead_time)]
        self.weights_lead_time_label = np.zeros((self.lead_time, self.n_classes))
        self.seen_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))
        self.correct_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))
        self.iou_de_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))
        self.predicted_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))

    def __call__(self, pred, target, data_glofas, thresholds): # the threshold is not the same as target var -- we should convert back it to original value by adding data_glofas
        # pred: Batch, seq_length,feature(=1)
        # target: Batch, seq_length,feature(=1)
        # thresholds: Batch, 9
        # Here, no need to iterate over batch based lead time with shape (Batch,)
        # Rather, iterate over the lead time to calculate the metrics one by one
        # data_glofas: [Batch, 1] -- to take the last step of dis24 from training script

        self.seen_iter += 1
        '''
        
        mask = ~np.isnan(target)  # Create mask where y is not NaN

        # Apply mask to both x and y

        target_filtered = target[mask]
        pred_filtered = pred[mask]

        print (mask.shape)
        print (target.shape)
        print (target_filtered.shape)
        print (data_glofas.shape)
        print (data_glofas.reshape(-1, 1, 1).shape)
        data_glofas_filtered = data_glofas.reshape(-1, 1, 1)[mask]
        '''
        
        target_convert = target + data_glofas.reshape(-1, 1, 1)
        pred_convert = pred + data_glofas.reshape(-1, 1, 1)

        mask = ~np.isnan(target)
        target_convert = np.where(mask, target + data_glofas.reshape(-1, 1, 1), np.nan)
        pred_convert = np.where(mask, pred + data_glofas.reshape(-1, 1, 1), np.nan)

        # we should use the converted values to calculate the metrics by adding up the baseline

        self.dis_mae += np.nanmean(np.abs(pred_convert - target_convert))
        self.dis_rmse += np.sqrt(np.nanmean((pred_convert - target_convert) ** 2))

        
        self.dis_r2 += safe_metric(r2_score, pred_convert, target_convert)
        self.dis_kge += safe_metric(kge_multi, pred_convert, target_convert)
        self.dis_pearson_r += safe_metric(pearson_r, pred_convert, target_convert)
        self.dis_variability_ratio += safe_metric(variability_ratio, pred_convert, target_convert)
        self.dis_bias_ratio += safe_metric(bias_ratio, pred_convert, target_convert)

        thresholds_expanded = thresholds[:, np.newaxis, :]  # Shape becomes (1280, 1, 9)
        pred_thr = (pred_convert >= thresholds_expanded)  # Shape will be (1280, 10, 9)
        target_thr = (target_convert >= thresholds_expanded)  # Shape will be (1280, 10, 9)

        self.seen_all += np.prod(target_thr[:, :, 0].shape)
        self.weights_label += np.sum(target_thr, axis=(0, 1))

        for label in range(self.n_classes):
            self.correct_label_all[label] += np.sum((pred_thr[:, :, label] == 1) & (target_thr[:, :, label] == 1))
            self.seen_label_all[label] += np.sum((target_thr[:, :, label] == 1))
            self.iou_de_label_all[label] += np.sum(((pred_thr[:, :, label] == 1) | (target_thr[:, :, label] == 1)))
            self.predicted_label_all[label] += np.sum(pred_thr[:, :, label] == 1)

        for t in range(self.lead_time):  # Iterate over all lead times
            pred_t = pred_convert[:, t, :]  # Extract predictions for lead time t
            target_t = target_convert[:, t, :]  # Extract targets for lead time t

            self.dis_lead_time_mae[t] += np.nanmean(np.abs(pred_t - target_t))
            self.dis_lead_time_rmse[t] += np.sqrt(np.nanmean((pred_t - target_t) ** 2))

            # 用 safe_metric 屏蔽 NaN
            self.dis_lead_time_r2[t] += safe_metric(r2_score, pred_t, target_t)
            self.dis_lead_time_kge[t] += safe_metric(kge_multi, pred_t, target_t)
            self.dis_lead_time_pearson_r[t] += safe_metric(pearson_r, pred_t, target_t)
            self.dis_lead_time_variability_ratio[t] += safe_metric(variability_ratio, pred_t, target_t)
            self.dis_lead_time_bias_ratio[t] += safe_metric(bias_ratio, pred_t, target_t)

            '''
            self.dis_lead_time_mae[t] += np.mean(np.abs(pred_t - target_t))
            self.dis_lead_time_rmse[t] += np.sqrt(np.mean((pred_t - target_t) ** 2))
            self.dis_lead_time_r2[t] += r2_score_multi(pred_t.flatten(), target_t.flatten())
            self.dis_lead_time_kge[t] += kge_multi(pred_t.flatten(), target_t.flatten())
            self.dis_lead_time_pearson_r[t] += pearson_r(pred_t.flatten(), target_t.flatten())
            self.dis_lead_time_variability_ratio[t] += variability_ratio(pred_t.flatten(), target_t.flatten())
            self.dis_lead_time_bias_ratio[t] += bias_ratio(pred_t.flatten(), target_t.flatten())
            
            '''
            
            self.seen_lead_time_iter[t] += 1 

            self.seen_lead_time_all[t] += np.prod(target_thr[:, t, 0].shape)
            self.weights_lead_time_label[t, :] += np.sum(target_thr[:, t, :], axis=0)

            for label in range(self.n_classes):
                self.correct_lead_time_label_all[t, label] += np.sum((pred_thr[:, t, label] == 1) & (target_thr[:, t, label] == 1))
                self.seen_lead_time_label_all[t, label] += np.sum((target_thr[:, t, label] == 1))
                self.iou_de_lead_time_label_all[t, label] += np.sum(((pred_thr[:, t, label] == 1) | (target_thr[:, t, label] == 1)))
                self.predicted_lead_time_label_all[t, label] += np.sum(pred_thr[:, t, label] == 1)


def min_max_scale(array, min_array, max_array, min_new=-1., max_new=1.):
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


def save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, metric):
    dir_log = os.path.join(config.dir_log, config.name)
    checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
    if metric == 'loss':
        path = os.path.join(checkpoints_dir, 'best_loss_model.pth')
        log_string(logger, 'saving model to %s' % path)

    elif metric == 'F1':
        path = os.path.join(checkpoints_dir, 'best_F1_model.pth')
        log_string(logger, 'saving model to %s' % path)

    elif metric == 'R2':
        path = os.path.join(checkpoints_dir, 'best_R2_model.pth')
        log_string(logger, 'saving model to %s' % path)

    elif metric == 'train':
        path = os.path.join(checkpoints_dir, 'best_train_model.pth')

    elif metric == 'latest':
        path = os.path.join(checkpoints_dir, 'latest_model.pth')

    state = {
        'epoch': epoch,
        'mean_loss_train': mean_loss_train,
        'mean_loss_validation': mean_loss_val,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(state, path)


def generate_images(pred, target, data_thresholds, mask_valid, height, width, lead_time):

    B, _ = pred.shape
    im_pred = np.repeat(mask_valid[np.newaxis, :], B, axis=0)
    im_pred[:, mask_valid == 0] = np.nan
    im_pred[:, mask_valid == 1] = pred
    im_pred = im_pred.reshape(B, height, width)

    im_target = np.repeat(mask_valid[np.newaxis, :], B, axis=0)
    im_target[:, mask_valid == 0] = np.nan
    im_target[:, mask_valid == 1] = target
    im_target = im_target.reshape(B, height, width)

    im_diff = im_pred - im_target

    fig_diff = plt.figure()
    for idx in np.arange(B):
        ax = fig_diff.add_subplot(1, B, idx+1, xticks=[], yticks=[])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.1)
        v_max = np.nanmax(np.abs(im_diff[idx, :, :]))
        im = ax.imshow(im_diff[idx, :, :], cmap='BrBG', vmin=-v_max, vmax=v_max)
        ax.axis('off')
        fig_diff.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title('lead time: ' + str(lead_time[idx]))
    fig_diff.tight_layout()
    #plt.show()

    # data_thresholds B, P, 9
    im_thresholds = np.repeat(mask_valid[np.newaxis, :], B, axis=0)
    im_thresholds = np.repeat(im_thresholds[:, :, np.newaxis], 9, axis=-1)
    im_thresholds[:, mask_valid == 0, :] = np.nan
    for thr in range(9):
        im_thresholds[:, mask_valid == 1, thr] = data_thresholds[:, :, thr]
    im_thresholds = im_thresholds.reshape(B, height, width, 9)

    im_pred_thr = (np.repeat(im_pred[:, :, :, np.newaxis], 9, axis=-1) >= im_thresholds).astype(np.float32)
    im_target_thr = (np.repeat(im_target[:, :, :, np.newaxis], 9, axis=-1) >= im_thresholds).astype(np.float32)

    im_pred_thr[:, mask_valid.reshape(height, width) == 0, :] = np.nan
    im_target_thr[:, mask_valid.reshape(height, width) == 0, :] = np.nan

    classes = ["1.5", "2  ", "5  ", "10 ", "20 ", "50 ", "100", "200", "500"]

    fig_thr = []
    for f in range(9):
        fig, axs = plt.subplots(2, B)
        for idx in np.arange(B):
            if B > 1:
                ax = axs[0, idx]
            else:
                ax = axs[0]
            ax.axis('off')
            ax.imshow(im_pred_thr[idx, :, :, f], vmin=0, vmax=1)
            ax.set_title('pred thr' + classes[f])
            if B > 1:
                ax = axs[1, idx]
            else:
                ax = axs[1]
            ax.axis('off')
            ax.imshow(im_target_thr[idx, :, :, f], vmin=0, vmax=1)
            ax.set_title('target thr ' + classes[f])

        fig.tight_layout()
        fig_thr.append(fig)

    return fig_diff, fig_thr


if __name__ == '__main__':

   print()