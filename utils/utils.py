# ------------------------------------------------------------------
"""
Utility functions

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import numpy as np
import random
import os
import datetime
import logging
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import r2_score
from timm.optim import optim_factory
from types import SimpleNamespace

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

# ------------------------------------------------------------------


def log_string(logger, str):
    logger.info(str)
    print(str)


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


def get_optimizer(model, config):

    args1 = SimpleNamespace()
    args1.weight_decay = config.weight_decay
    args1.lr = config.lr
    args1.opt = config.optimizer
    args1.momentum = config.beta
    optimizer = optim_factory.create_optimizer(args1, model)

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

        self.seen_lead_time_iter = [0 for _ in range(lead_time)]
        self.dis_lead_time_mae = [0 for _ in range(lead_time)]
        self.dis_lead_time_rmse = [0 for _ in range(lead_time)]
        self.dis_lead_time_r2 = [0 for _ in range(lead_time)]
        self.dis_lead_time_kge = [0 for _ in range(lead_time)]

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
            message += 'lead time %s - %s   MAE: %.6f, RMSE: %.6f, R2: %.4f, KGE: %.4f \n' % (
                self.lead_time_classes_t[t],
                self.lead_time_classes[t] + ' ' * (7 - len(self.lead_time_classes[t])),
                self.dis_lead_time_mae[t],
                self.dis_lead_time_rmse[t],
                self.dis_lead_time_r2[t],
                self.dis_lead_time_kge[t]
            )

        message += '\n'

        #if self.lead_time > 1:
        self.dis_mae = self.dis_mae / float(self.seen_iter)
        self.dis_rmse = self.dis_rmse / float(self.seen_iter)
        self.dis_r2 = self.dis_r2 / float(self.seen_iter)
        self.dis_kge = self.dis_kge / float(self.seen_iter)

        message += 'lead time overall        MAE: %.6f, RMSE: %.6f, R2: %.4f, KGE: %.4f \n' % (
            self.dis_mae,
            self.dis_rmse,
            self.dis_r2,
            self.dis_kge)

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

        self.seen_lead_time_iter = [0 for _ in range(self.lead_time)]
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

        self.seen_lead_time_all = [0 for _ in range(self.lead_time)]
        self.weights_lead_time_label = np.zeros((self.lead_time, self.n_classes))
        self.seen_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))
        self.correct_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))
        self.iou_de_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))
        self.predicted_lead_time_label_all = np.zeros((self.lead_time, self.n_classes))

    def __call__(self, pred, target, thresholds, lead_time):

        # pred B, P, C
        # target B, P, C
        # thresholds B, P, 9
        # lead_time B

        #if self.lead_time > 1:
        # compute average regression scores
        self.dis_mae += np.mean(np.abs(pred - target))
        self.dis_rmse += np.sqrt(np.mean((pred - target) ** 2))
        self.dis_r2 += r2_score_multi(pred.flatten(), target.flatten())
        self.dis_kge += kge_multi(pred.flatten(), target.flatten())

        self.seen_iter += 1

        pred_thr = (np.repeat(pred[:, :, 0:1], 9, axis=-1) >= thresholds)
        target_thr = (np.repeat(target[:, :, 0:1], 9, axis=-1) >= thresholds)

        # compute average classification scores
        self.seen_all += np.prod(target_thr[:, :, 0].shape)
        self.weights_label += np.sum(target_thr, axis=(0, 1))

        for label in range(self.n_classes):
            self.correct_label_all[label] += np.sum((pred_thr[:, :, label] == 1) & (target_thr[:, :, label] == 1))
            self.seen_label_all[label] += np.sum((target_thr[:, :, label] == 1))
            self.iou_de_label_all[label] += np.sum(((pred_thr[:, :, label] == 1) | (target_thr[:, :, label] == 1)))
            self.predicted_label_all[label] += np.sum(pred_thr[:, :, label] == 1)

        # compute per lead time regression scores
        for b in range(len(lead_time)):
            self.dis_lead_time_mae[lead_time[b] - 1] += np.mean(np.abs(pred[b, :, :] - target[b, :, :]))
            self.dis_lead_time_rmse[lead_time[b] - 1] += np.sqrt(np.mean((pred[b, :, :] - target[b, :, :]) ** 2))
            self.dis_lead_time_r2[lead_time[b] - 1] += r2_score_multi(pred[b, :, :].flatten(), target[b, :, :].flatten())
            self.dis_lead_time_kge[lead_time[b] - 1] += kge_multi(pred[b, :, :].flatten(), target[b, :, :].flatten())

            self.seen_lead_time_iter[lead_time[b] - 1] += 1
            # compute per lead time classification scores

            self.seen_lead_time_all[lead_time[b] - 1] += np.prod(target_thr[b, :, 0].shape)
            self.weights_lead_time_label[lead_time[b] - 1, :] += np.sum(target_thr[b, :, :], axis=0)

            for label in range(self.n_classes):
                self.correct_lead_time_label_all[lead_time[b] - 1, label] += np.sum((pred_thr[b, :, label] == 1) & (target_thr[b, :, label] == 1))
                self.seen_lead_time_label_all[lead_time[b] - 1, label] += np.sum((target_thr[b, :, label] == 1))
                self.iou_de_lead_time_label_all[lead_time[b] - 1, label] += np.sum(((pred_thr[b, :, label] == 1) | (target_thr[b, :, label] == 1)))
                self.predicted_lead_time_label_all[lead_time[b] - 1, label] += np.sum(pred_thr[b, :, label] == 1)



def min_max_scale(array, min_array, max_array, min_new=-1., max_new=1.):
    """
        normalize an array between new minimum and maximum values

        Parameters
        ----------
        array : numpy array
           array to be normalized
        min_alt : float
           minimum value in array
        max_alt : float
           maximum value in array
        min_new : float
           minimum value after normalization
        max_new : float
           maximum value after normalization

        Returns
        ----------
        array : numpy array
           normalized numpy array
    """
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


def save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, best_loss_train, best_loss_val, logger, config, metric):
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

    state = {
        'epoch': epoch,
        'mean_loss_train': mean_loss_train,
        'mean_loss_validation': mean_loss_val,
        'best_loss_train': best_loss_train,
        'best_loss_validation': best_loss_val,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, path)

