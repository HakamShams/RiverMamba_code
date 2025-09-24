# ------------------------------------------------------------------
# Main config file
# ------------------------------------------------------------------
from typing import Dict, List, Optional
import argparse
import pickle
import os
import datetime


# ------------------------------------------------------------------

def add_all_arguments(parser):

    # --- general options --- #
    parser.add_argument('--name', type=str, default='edbilstm_obsmap_era5tar_obstar_weight_14', help='name of the experiment')
    parser.add_argument('--dir_log', type=str, default=r'./RiverMamba_code/lstm_repo/log', help='log folder')
    
    # finetune and resume can only be one True and one False
    parser.add_argument('--finetune', type=bool, default=True, help='fine tune on obs or not')
    parser.add_argument('--pretrained_model_path', type=str, default=r'./RiverMamba_code/lstm_repo/log/edbilstm_obsmap_era5tar_weight_14/model_checkpoints/best_loss_model.pth', help='pretrained_model_path from era5 reanalysis')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from latest checkpoint')
    parser.add_argument('--dir_model', type=str, default=r'./RiverMamba_code/lstm_repo/log/edbilstm_obsmap_era5tar_obstar_weight_14/model_checkpoints/', help='Directory to load model checkpoints and resume training (the model run itself)')

    parser.add_argument('--is_hres_forecast', type=bool, default=True, help='use hres forecast as decoder input or not')
    parser.add_argument('--is_target_obs', type=bool, default=True, help='use GRDC obs as target for loss and eval or not')
    parser.add_argument('--use_weighted_loss', type=bool, default=True, help='use weighted loss or not')
    ###### choose the name of the model type ########### maybe set later TO DO
    #parser.add_argument('--encoder', type=str, default='mutlihead_lstm', help='name of the encoder model') ### mutlihead_lstm

    # --- lstm common para ---
    parser.add_argument('--output_dropout', type=float, default=0.4, help='dropout after LSTM layer')
    parser.add_argument('--output_size', type=int, default=1, help='number of target variables')
    #parser.add_argument('--forecast_seq_length', type=int, default=5, help='length of forecast lead time')
    parser.add_argument('--static_embedding_spec', type=Dict, default={'type': 'fc','hiddens': [20], 'activation': 'tanh', 'dropout': 0}, help='parameter for static var embedding')
    parser.add_argument('--dynamic_embedding_spec', type=Dict, default={'type': 'fc','hiddens': [20], 'activation': 'tanh', 'dropout': 0}, help='parameter for dynamic var embedding')
    parser.add_argument('--head_type', type=str, default='regression', help='name of the forecast head model')
    parser.add_argument('--initial_forget_bias', type=int, default=3, help='Initial bias value of the forget gate')

    # --- multihead_lstm ---
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden dimension for the hindcastLSTM')
    parser.add_argument('--forecast_network_para', type=Dict, default={'hiddens': [128, 64], 'activation': 'relu', 'dropout': 0.1}, help='parameter for FC forecast layer')

    # --- ed_bilstm ---
    parser.add_argument('--hindcast_hidden_size', type=int, default=256, help='hidden dimension for the hindcast bidirectional LSTM')
    parser.add_argument('--forecast_hidden_size', type=int, default=128, help='hidden dimension for the forecastLSTM')
    parser.add_argument('--state_handoff_network_para', type=Dict, default={'hiddens': [128], 'activation': 'tanh', 'dropout': 0.1}, help='parameter for state_handoff_network layer')
    # hidden size here includes both forward layer and output layer

    # --- dataset class --- #
    parser.add_argument('--root_glofas_reanalysis', type=str,
                        default=r'./RiverMamba_dataset/GRDC_masked_dataset/GloFAS_Reanalysis', help='log folder')
    parser.add_argument('--root_era5_land_reanalysis', type=str,
                        default=r'./RiverMamba_dataset/GRDC_masked_dataset/ERA5-Land_Reanalysis_Global', help='log folder')
    parser.add_argument('--root_hres_forecast', type=str,
                        default=r'./RiverMamba_dataset/GRDC_masked_dataset/ECMWF_HRES_global', help='log folder')
    parser.add_argument('--root_static', type=str,
                        default=r'./RiverMamba_dataset/GRDC_masked_dataset/GloFAS_Static', help='log folder')
    parser.add_argument('--root_cpc', type=str,
                        default=r'./RiverMamba_dataset/GRDC_masked_dataset/CPC_regridded_nearest', help='log folder')
    parser.add_argument('--root_obs', type=str,
                        default=r'./RiverMamba_dataset/GRDC_masked_dataset/GRDC_Obs', help='log folder')


    parser.add_argument('--years_train', type=str, default=[str(year) for year in range(1979, 2019)], help='years for training')
    parser.add_argument('--years_val', type=str, default=['2019','2020'], help='years for validation')
    parser.add_argument('--years_test', type=str, default=['2019','2020','2021','2022','2023','2024'], help='years for testing')

    parser.add_argument('--delta_t', type=int, default=14, help='length of input')
    parser.add_argument('--delta_t_f', type=int, default=7, help='forecast lead time, maximum 7 days') 
    parser.add_argument('--is_random_t', type=bool, default=False, help='random selecting number of weeks or not')

    
    parser.add_argument('--n_points', type=int, default=3366, help='fixed num for total stations') # number of grid points selecting from one map
    parser.add_argument('--lat_min', type=int, default=-90, help='')
    parser.add_argument('--lat_max', type=int, default=90, help='')  # 1 / 0.05 = 20 pixels >> 20 * 30 = 600 height
    parser.add_argument('--lon_min', type=int, default=-180, help='')
    parser.add_argument('--lon_max', type=int, default=180, help='')  # 1000 width
    parser.add_argument('--nan_fill', type=float, default=0., help='a value to fill missing values')
    parser.add_argument('--is_shuffle', type=bool, default=False, help='if True, apply data shuffling')
    parser.add_argument('--is_aug', type=bool, default=False, help='if True, apply data augmentation')
    parser.add_argument('--is_norm', type=bool, default=True, help='if True, apply data normalization')

    # --- training parameters --- #
    parser.add_argument('--gpu_id', type=str, default="0", help='gpu ids: i.e. 0  (0,1,2, use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_workers_train', type=int, default=16, help='number of workers for multiprocessing')
    parser.add_argument('--n_workers_val', type=int, default=16, help='number of workers for multiprocessing')

    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='allocate the loaded samples in GPU memory. Use it with training on GPU')

    parser.add_argument('--batch_size_train', type=int, default=1, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 momentum term for Adam/AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 momentum term for Adam/AdamW')

    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler')
    parser.add_argument('--lr_warmup', type=int, default=1e-6, help='learning rate for warmup')
    parser.add_argument('--lr_warmup_epochs', type=int, default=3, help='number of epochs for warmup')
    parser.add_argument('--lr_min', type=float, default=0.000001, help='minimum learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate step decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.3, help='learning rate decay')

    # early stopping
    parser.add_argument('--patience', type=int, default=6, help='patience for early stopping')

    #parser.add_argument('--lambda', type=float, default=10., help='weight')

    # --- input variables --- #
    parser.add_argument('--in_glofas', type=int, default=4, help='')
    parser.add_argument('--in_era5', type=int, default=24-6, help='')
    parser.add_argument('--in_static', type=int, default=103, help='')
    parser.add_argument('--in_hres', type=int, default=7, help='')

    parser.add_argument('--variables_glofas', type=str,
                        default=[
                            'acc_rod24',
                            'dis24',
                            'sd',
                            'swi'
                        ],
                        help='input variables_glofas')

    parser.add_argument('--variables_era5_land', type=str,
                        default=['d2m', 'e', 'es', 'evabs', 'evaow', 'evatc', 'evavt', 'lai_hv', 'lai_lv', 'pev', 'sf',
                           'skt', 'slhf', 'smlt', 'sp', 'src', 'sro', 'sshf', 'ssr', 'ssrd', 'ssro',
                           'stl1', 'stl2', 'stl3', 'stl4', 'str', 'strd', 'swvl1', 'swvl2', 'swvl3', 'swvl4',
                           't2m', 'tp', 'u10', 'v10'
                           ],
                        help='input variables_era5_land')
    
    # only for hres forecast
    parser.add_argument('--variables_hres_forecast', type=str,
                        default=['e', 'sf', 'sp', 'ssr', 'str', 't2m', 'tp'],
                        help='input variables_hres_forecast')

    parser.add_argument('--variables_static', type=str,
                        default=[
                            'CalChanMan1', 'CalChanMan2', 'GwLoss', 'GwPercValue', 'LZTC', 'LZthreshold',
                             'LakeMultiplier', 'PowerPrefFlow', 'QSplitMult', 'ReservoirRnormqMult', 'SnowMeltCoef',
                             'UZTC', 'adjustNormalFlood', 'b_Xinanjiang', 'chan', 'chanbnkf', 'chanbw', 'chanflpn',
                             'changrad', 'chanlength', 'chanman', 'chans', 'cropcoef', 'cropgrpn', 'elv', 'elvstd',
                             'enconsuse', 'fracforest', 'fracgwused', 'fracirrigated', 'fracncused', 'fracother',
                             'fracrice', 'fracsealed', 'fracwater', 'genua1', 'genua2', 'genua3', 'gradient',
                             'gwbodies', 'ksat1', 'ksat2', 'ksat3', 'laif_01', 'laif_02', 'laif_03', 'laif_04',
                             'laif_05', 'laif_06', 'laif_07', 'laif_08', 'laif_09', 'laif_10', 'laif_11', 'laif_12',
                             'laii_01', 'laii_02', 'laii_03', 'laii_04', 'laii_05', 'laii_06', 'laii_07', 'laii_08',
                             'laii_09', 'laii_10', 'laii_11', 'laii_12', 'laio_01', 'laio_02', 'laio_03', 'laio_04',
                             'laio_05', 'laio_06', 'laio_07', 'laio_08', 'laio_09', 'laio_10', 'laio_11', 'laio_12',
                             'lakea', 'lakearea', 'lakeavinflow', 'lambda1', 'lambda2', 'lambda3', 'ldd', 'lusemask',
                             'mannings', 'outlets', 'pixarea', 'pixleng', 'rclim', 'rflim_97th', 'riceharvestday1',
                             'riceharvestday2', 'riceharvestday3', 'riceplantingday1', 'riceplantingday2',
                             'riceplantingday3', 'rminq_5th', 'rndq_97th', 'rnlim_67th', 'rnormq_50th', 'rstor',
                             'soildepth1', 'soildepth2', 'soildepth3', 'thetar1', 'thetar2', 'thetar3', 'thetas1',
                             'thetas2', 'thetas3', 'upArea', 'waterregions'
                        ],
                       
                        help='input variables_static')

    # --- variables for log1p transformations --- # -- do we need to keep this
    parser.add_argument('--variables_glofas_log1p', type=str,
                        default=None,#[
                            #'acc_rod24',
                            #'dis24',
                            #'sd',
                            #'swi'
                        #],
                        help='glofas variables that need log1p transformation')

    parser.add_argument('--variables_era5_land_log1p', type=str,
                        default=None,
                        help='era5 land variables that need log1p transformation')

    parser.add_argument('--variables_hres_forecast_log1p', type=str,
                        default=None,
                        help='hres forecast variables that need log1p transformation')

    parser.add_argument('--variables_static_log1p', type=str,
                        default=[
                            "chanbw", "chanflpn", "elvstd", "ksat1", "ksat2", "ksat3", "soildepth2", "soildepth3",
                            "upArea", "waterregions"
                        ],
                        help='static variables that need log1p transformation')
    
    # --- inference vars --- #
    parser.add_argument('--output_path', type=str,
                        default=r'./RiverMamba_inference/', help='log folder')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r'./RiverMamba_code/lstm_repo/log', help='log folder')

    return parser


def read_arguments(train=True, print=True, save=True):
    # used to read arguments from 'add_all_arguments'
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    parser.add_argument('--phase', type=str, default='train')
    config = parser.parse_args()
    config.phase = 'train' if train else 'test'
    if print:
        print_options(config, parser)
    if save:
        save_options(config, parser)
    return config


def save_options(config, parser): 

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    os.makedirs(dir_log, exist_ok=True)

    with open(dir_log + '/config.txt', 'wt') as config_file:
        message = ''
        message += '----------------- Options ---------------       -------------------\n\n'
        for k, v in sorted(vars(config).items()):
            if k in ['variables_glofas', 'variables_era5_land', 'variables_static',
                     'variables_glofas_log1p', 'variables_era5_land_log1p', 'variables_static_log1p',
                     'years_train', 'years_val', 'years_test', 'dir_log',
                     'root_glofas_reanalysis', 'root_era5_land_reanalysis', 'root_static']:
                continue
            # comment = ''
            default = parser.get_default(k)
            # if v != default:
            comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

        comment = '\t[default: %s]' % str(parser.get_default('root_glofas_reanalysis'))
        message += '\n{:>25}: {:<20}{}\n'.format('root_glofas_reanalysis',
                                                 vars(config)['root_glofas_reanalysis'], comment)
        comment = '\t[default: %s]' % str(parser.get_default('root_era5_land_reanalysis'))
        message += '{:>25}: {:<20}{}\n'.format('root_era5_land_reanalysis',
                                                 vars(config)['root_era5_land_reanalysis'], comment)
        comment = '\t[default: %s]' % str(parser.get_default('root_static'))
        message += '{:>25}: {:<20}{}\n'.format('root_static',
                                                 vars(config)['root_static'], comment)

        comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
        message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

        message += '\n----------------- Input Variables -------      -------------------'
        message += '\n\n{:>20}: {}\n'.format('Variables GloFAS', str(config.variables_glofas))
        message += '{:>20}: {}\n'.format('Variables Era5-Land', str(config.variables_era5_land))
        message += '{:>20}: {}\n'.format('Variables Static', str(config.variables_static))

        message += '\n{:>26}: {}\n'.format('Variables GloFAS log1p', str(config.variables_glofas_log1p))
        message += '{:>26}: {}\n'.format('Variables Era5-Land logp1', str(config.variables_era5_land_log1p))
        message += '{:>26}: {}\n'.format('Variables Static logp1', str(config.variables_static_log1p))

        message += '\n----------------- Years -----------------      -------------------'
        if config.phase == 'train':
            message += '\n\n{:>20}: {}'.format('Training', str(config.years_train))
            message += '\n{:>20}: {}\n'.format('Validation', str(config.years_val))
        else:
            message += '\n\n{:>20}: {}\n'.format('Testing', str(config.years_test))

        message += '\n----------------- End -------------------      -------------------'
        config_file.write(message)

    with open(dir_log + '/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)


def print_options(config, parser):
    message = ''
    message += '----------------- Options ---------------       -------------------\n\n'
    for k, v in sorted(vars(config).items()):
        if k in ['variables_glofas', 'variables_era5_land', 'variables_static',
                 'variables_glofas_log1p', 'variables_era5_land_log1p', 'variables_static_log1p',
                 'years_train', 'years_val', 'years_test', 'dir_log',
                 'root_glofas_reanalysis', 'root_era5_land_reanalysis', 'root_static']:
            continue
        # comment = ''
        default = parser.get_default(k)
        # if v != default:
        comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

    comment = '\t[default: %s]' % str(parser.get_default('root_glofas_reanalysis'))
    message += '\n{:>25}: {:<20}{}\n'.format('root_glofas_reanalysis',
                                             vars(config)['root_glofas_reanalysis'], comment)
    comment = '\t[default: %s]' % str(parser.get_default('root_era5_land_reanalysis'))
    message += '{:>25}: {:<20}{}\n'.format('root_era5_land_reanalysis',
                                             vars(config)['root_era5_land_reanalysis'], comment)
    comment = '\t[default: %s]' % str(parser.get_default('root_static'))
    message += '{:>25}: {:<20}{}\n'.format('root_static',
                                             vars(config)['root_static'], comment)

    comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
    message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

    message += '\n----------------- Input Variables -------      -------------------'
    message += '\n\n{:>20}: {}\n'.format('Variables GloFAS', str(config.variables_glofas))
    message += '{:>20}: {}\n'.format('Variables Era5-Land', str(config.variables_era5_land))
    message += '{:>20}: {}\n'.format('Variables Static', str(config.variables_static))

    message += '\n{:>26}: {}\n'.format('Variables GloFAS log1p', str(config.variables_glofas_log1p))
    message += '{:>26}: {}\n'.format('Variables Era5-Land logp1', str(config.variables_era5_land_log1p))
    message += '{:>26}: {}\n'.format('Variables Static logp1', str(config.variables_static_log1p))

    message += '\n----------------- Years -----------------      -------------------'
    if config.phase == 'train':
        message += '\n\n{:>20}: {}'.format('Training', str(config.years_train))
        message += '\n{:>20}: {}\n'.format('Validation', str(config.years_val))
    else:
        message += '\n\n{:>20}: {}\n'.format('Testing', str(config.years_test))

    message += '\n----------------- End -------------------      -------------------'
    print(message)


if __name__ == '__main__':

    config = read_arguments(train=True, print=True, save=False)


