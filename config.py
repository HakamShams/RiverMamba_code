# ------------------------------------------------------------------
"""
Main config file for RiverMamba

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import argparse
import pickle
import os
import datetime

# ------------------------------------------------------------------


def add_all_arguments(parser):

    # --- general options --- #
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--dir_log', type=str, default=r'./log', help='log folder')

    parser.add_argument('--encoder', type=str, default='Mamba', help='name of the encoder model')
    parser.add_argument('--decoder', type=str, default='Mamba', help='name of the decoder model')
    parser.add_argument('--head', type=str, default='MLP', help='name of the regression head model')

    # --- encoder --- #
    parser.add_argument('--en_embed_glofas', type=int, default=64-16, help='embedding dimension for GloFAS reanalysis')
    parser.add_argument('--en_embed_era5', type=int, default=128, help='embedding dimension for ERA5-Land reanalysis')
    parser.add_argument('--en_embed_cpc', type=int, default=16, help='embedding dimension for CPC precipitation')

    parser.add_argument('--en_embed_dim', type=int, default=[192, 192, 192],
                        help='embedding dimensions in the encoder model')
    parser.add_argument('--en_depths', type=int, default=[2, 2, 2], help='number of blocks inside each layer')
    parser.add_argument('--en_grouping_size', type=int, default=[(4, 254945), (2, 254945), (1, 254945)],
                        help='grouping size for mamba/self-attention')
    parser.add_argument('--en_mlp_ratio', type=float, default=1., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--en_drop_rate', type=float, default=0., help='dropout rate')
    parser.add_argument('--en_drop_path_rate', type=float, default=0., help='stochastic depth rate')
    parser.add_argument('--en_embed_norm', type=bool, default=True,
                        help='if True, add normalization after point embedding')
    parser.add_argument('--en_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--en_use_reentrant', type=bool, default=True, help='whether to use reentrant for checkpoint')

    parser.add_argument('--en_curve_order', type=str, default='spatial_first',
                        help='order of the curve [temporal, spatial, spatial_first, temporal_first]')

    # encoder FlashAttention
    parser.add_argument('--en_n_heads', type=int, default=[1, 1, 1], help='number of heads for self-attention')
    parser.add_argument('--en_attn_drop_rate', type=float, default=0.0, help='attention dropout rate')
    parser.add_argument('--en_qkv_bias', type=bool, default=False,
                        help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--en_proj_drop', type=float, default=0., help='projection dropout rate after attention')
    parser.add_argument('--en_qk_scale', type=float, default=None,
                        help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--en_is_causal', type=bool, default=False, help='whether to use causal attention')
    parser.add_argument('--en_is_alibi_slopes', type=bool, default=False,
                        help='whether to use alibi encoding for self-attention')
    parser.add_argument('--en_att_window_size', type=int, default=None, help='attention window size')

    # encoder Mamba
    parser.add_argument('--en_d_state', type=int, default=[1, 1, 1], help='SSM state expansion factor')
    parser.add_argument('--en_d_conv', type=int, default=[3, 3, 3], help='local convolution width')
    parser.add_argument('--en_expand', type=int, default=[1, 1, 1], help='d_inner expansion factor')
    parser.add_argument('--en_dt_min', type=int, default=0.001, help='SSM dt_min')
    parser.add_argument('--en_dt_max', type=int, default=0.1, help='SSM dt_max')
    parser.add_argument('--en_dt_rank', type=str, default='auto', help='SSM dt_rank')
    parser.add_argument('--en_dt_init', type=str, default='random', help='SSM dt_init')
    parser.add_argument('--en_dt_scale', type=float, default=1.0, help='SSM dt_scale')
    parser.add_argument('--en_dt_init_floor', type=float, default=1e-4, help='SSM dt_init_floor')

    parser.add_argument('--en_conv_bias', type=bool, default=False,
                        help='whether to use bias in the convolution of SSM')
    parser.add_argument('--en_proj_bias', type=bool, default=False,
                        help='whether to use bias in the projection of SSM')
    parser.add_argument('--en_is_divide_out', type=bool, default=True,
                        help='whether to divide bidirectional SSM by 2 before the output')
    parser.add_argument('--en_is_out_norm', type=bool, default=False,
                        help='whether to use normalization after at the end of SSM')

    parser.add_argument('--en_bi_ssm', type=bool, default=True, help='whether to use bidirectional SSM')

    # for Mamba2
    parser.add_argument('--en_ngroups', type=int, default=1, help='number of groups in each layer')
    parser.add_argument('--en_headdim', type=int, default=64, help='head dimension in each layer')
    parser.add_argument('--en_chunk_size', type=int, default=256, help='chunk size')

    # --- decoder --- #
    parser.add_argument('--de_embed_hres', type=int, default=64, help='embedding dimension for ECMWF-HRES')
    parser.add_argument('--de_embed_dim', type=int, default=[192], help='embedding dimensions in the decoder model')
    parser.add_argument('--de_depths', type=int, default=[1], help='number of blocks inside each layer')
    parser.add_argument('--de_grouping_size', type=int, default=[(1, 254945)],
                        help='grouping size for mamba/self-attention')
    parser.add_argument('--de_mlp_ratio', type=float, default=1., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--de_drop_rate', type=float, default=0., help='dropout rate')
    parser.add_argument('--de_drop_path_rate', type=float, default=0., help='stochastic depth rate')
    parser.add_argument('--de_embed_norm', type=bool, default=True,
                        help='if True, add normalization after point embedding')
    parser.add_argument('--de_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--de_use_reentrant', type=bool, default=True,
                        help='whether to use reentrant for checkpoint')
    parser.add_argument('--de_curve_order', type=str, default='spatial_first',
                        help='order of the curve [temporal, spatial, spatial_first, temporal_first]')

    # decoder FlashAttention
    parser.add_argument('--de_n_heads', type=int, default=[1], help='number of heads for self-attention')
    parser.add_argument('--de_attn_drop_rate', type=float, default=0.0, help='attention dropout rate')
    parser.add_argument('--de_qkv_bias', type=bool, default=False,
                        help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--de_proj_drop', type=float, default=0., help='projection dropout rate after attention')
    parser.add_argument('--de_qk_scale', type=float, default=None,
                        help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--de_is_causal', type=bool, default=False, help='whether to use causal attention')
    parser.add_argument('--de_is_alibi_slopes', type=bool, default=False,
                        help='whether to use alibi encoding for self-attention')
    parser.add_argument('--de_att_window_size', type=int, default=None, help='whether to use attention window size')

    # decoder Mamba
    parser.add_argument('--de_d_state', type=int, default=[1], help='SSM state expansion factor')
    parser.add_argument('--de_d_conv', type=int, default=[3], help='local convolution width')
    parser.add_argument('--de_expand', type=int, default=[1], help='d_inner expansion factor')
    parser.add_argument('--de_dt_min', type=int, default=0.001, help='SSM dt_min')
    parser.add_argument('--de_dt_max', type=int, default=0.1, help='SSM dt_max')
    parser.add_argument('--de_dt_rank', type=str, default='auto', help='SSM dt_rank')
    parser.add_argument('--de_dt_init', type=str, default='random', help='SSM dt_init')
    parser.add_argument('--de_dt_scale', type=float, default=1.0, help='SSM dt_scale')
    parser.add_argument('--de_dt_init_floor', type=float, default=1e-4, help='SSM dt_init_floor')
    parser.add_argument('--de_conv_bias', type=bool, default=False,
                        help='whether to use bias in the convolution of SSM')
    parser.add_argument('--de_proj_bias', type=bool, default=False,
                        help='whether to use bias in the projection of SSM')
    parser.add_argument('--de_is_divide_out', type=bool, default=True,
                        help='whether to divide bidirectional SSM by 2 before the output')
    parser.add_argument('--de_is_out_norm', type=bool, default=False,
                        help='whether to use normalization after at the end of SSM')

    parser.add_argument('--de_bi_ssm', type=bool, default=True, help='whether to use bias in the convolution of SSM')

    parser.add_argument('--de_ngroups', type=int, default=1, help='number of groups in each layer')
    parser.add_argument('--de_headdim', type=int, default=8, help='head dimension in each layer')
    parser.add_argument('--de_chunk_size', type=int, default=256, help='chunk size')

    # --- regression head ---
    parser.add_argument('--head_hidden_dim', type=int, default=64, help='hidden dimension for the regression head')
    parser.add_argument('--head_out_dim', type=int, default=1, help='output dimension for the regression head')
    parser.add_argument('--head_drop_rate', type=float, default=0., help='head dropout rate')
    parser.add_argument('--head_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--head_use_reentrant', type=bool, default=True, help='whether to use reentrant for checkpoint')

    parser.add_argument('--pretrained_model', type=str,
                        #default=r'./log/test/model_checkpoints/best_loss_model.pth',
                        help='pretrained model i.e. trained model with best loss')

    parser.add_argument('--training_checkpoint', type=str,
                        #default=r'./log/test/model_checkpoints/best_train_model.pth',
                        help='training checkpoint to resume training')

    # --- dataset class --- #
    parser.add_argument('--root_glofas_reanalysis', type=str,
                        default=r'/home/ssd4tb/shams/GloFAS_Reanalysis_Global', help='root to GloFAS reanalysis')
    parser.add_argument('--root_era5_land_reanalysis', type=str,
                        default=r'/home/ssd4tb/shams/ERA5-Land_Reanalysis_Global', help='root to ERA5 reanalysis')
    parser.add_argument('--root_static', type=str,
                        default=r'/home/ssd4tb/shams/GloFAS_Static', help='root to GloFAS static')
    parser.add_argument('--root_hres_forecast', type=str,
                        default=r'/home/ssd4tb/shams/ECMWF_HRES_Global', help='root to ECMWF-HRES forecast')
    parser.add_argument('--root_cpc', type=str,
                        default=r'/home/ssd4tb/shams/CPC_Global', help='root to CPC precipitation')
    parser.add_argument('--root_obs', type=str,
                        default=r'/home/ssd4tb/shams/GRDC_Obs_Global', help='root to GRDC observations')

    parser.add_argument('--years_train', type=str, default=[str(year) for year in range(2024, 2024+1)], help='years for training')
    parser.add_argument('--years_val', type=str, default=['2024'], help='years for validation')
    parser.add_argument('--years_test', type=str, default=['2021', '2022', '2023', '2024'], help='years for testing')

    parser.add_argument('--delta_t', type=int, default=4,
                        help='number of days in the hindcast for the initial condition')
    parser.add_argument('--delta_t_f', type=int, default=7, help='lead time')
    parser.add_argument('--is_hres_forecast', type=bool, default=True,
                        help='whether to use ECMWF-HRES forecast otherwise ERA5 reanalysis will be used as forecast')

    parser.add_argument('--is_sample', type=bool, default=True, help='option to sample points')
    parser.add_argument('--is_sample_aifas', type=bool, default=True,
                        help='option to sample predefined AIFAS diagnostic river points')
    parser.add_argument('--is_sample_curves', type=bool, default=True,
                        help='option to sample points along the curve instead of random sampling')
    parser.add_argument('--n_points', type=int, default=254945, help='number of points')
    parser.add_argument('--is_obs', type=bool, default=False, help='whether to train with GRDC observations')
    parser.add_argument('--alpha', type=float, default=0.25, help='hyperparameter for the weight function')
    parser.add_argument('--static_dataset', type=str, default='LISFLOOD', help='name of the static dataset')

    parser.add_argument('--curves', type=str,
                        default=[
                            'sweep_h',
                            #'sweep_h_trans',
                            'sweep_v',
                            #'sweep_v_trans',
                            'gilbert',
                            'gilbert_trans',
                            #'zigzag_h',
                            #'zigzag_h_trans',
                            #'zigzag_v',
                            #'zigzag_v_trans'
                        ],
                        help='curves for serialization')
    parser.add_argument('--is_shuffle_curves', type=bool, default=False,
                        help='option to shuffle the curve in the dataloader')

    parser.add_argument('--lat_min', type=int, default=None, help='minimum latitude')
    parser.add_argument('--lat_max', type=int, default=None, help='maximum latitude')
    parser.add_argument('--lon_min', type=int, default=None, help='minimum longitude')
    parser.add_argument('--lon_max', type=int, default=None, help='maximum longitude')
    parser.add_argument('--nan_fill', type=float, default=0., help='a value to fill missing values')
    parser.add_argument('--is_shuffle', type=bool, default=False, help='if True, apply data samples shuffling')
    parser.add_argument('--is_norm', type=bool, default=True, help='if True, apply data normalization')

    # --- training parameters --- #
    parser.add_argument('--gpu_id', type=str, default="0", help='gpu ids: i.e. 0  (0,1,2, use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_workers_train', type=int, default=8, help='number of workers for training multiprocessing')
    parser.add_argument('--n_workers_val', type=int, default=8, help='number of workers for validation multiprocessing')

    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='allocate the loaded samples in GPU memory. Use it with training on GPU')

    parser.add_argument('--batch_size_train', type=int, default=1, help='batch size for training')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size for validation')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--val_every_n_epochs', type=int, default=1, help='validate every n epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='weight decay')
    parser.add_argument('--beta', type=float, default=0.9, help='beta momentum term for Adam/AdamW')
    parser.add_argument('--max_norm', type=float, default=10, help='maximum norm to clip gradient during training')

    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler')
    parser.add_argument('--lr_warmup', type=int, default=1e-6, help='learning rate for warmup')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='number of epochs for warmup')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate step decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='learning rate decay')

    # --- input variables --- #
    parser.add_argument('--is_add_xyz', type=bool, default=True,
                        help='option to add xyz WGS-84 coordinate to the static features')

    parser.add_argument('--in_glofas', type=int, default=4, help='number of GloFAS reanalysis variables')
    parser.add_argument('--in_hres', type=int, default=7, help='number of ECMWF-HRES meteorological forecast variables')
    parser.add_argument('--in_era5', type=int, default=35-3, help='number of ERA5-Land reanalysis variables')
    parser.add_argument('--in_static', type=int, default=96+3, help='number of static variables')

    parser.add_argument('--variables_glofas', type=str,
                        default=[
                            'acc_rod24',
                            'dis24',
                            'sd',
                            'swi'
                        ],
                        help='input variables_glofas')

    parser.add_argument('--variables_era5_land', type=str,
                        default=[
                            'd2m',
                            'e',
                            'es',
                            'evabs',
                            'evaow',
                            'evatc',
                            'evavt',
                            'lai_hv',
                            'lai_lv',
                            'pev',
                            'sf',
                            'skt',
                            'slhf',
                            'smlt',
                            'sp',
                            'src',
                            'sro',
                            'sshf',
                            'ssr',
                            'ssrd',
                            'ssro',
                            'stl1',
                            #'stl2',
                            #'stl3',
                            #'stl4',
                            'str',
                            'strd',
                            'swvl1',
                            'swvl2',
                            'swvl3',
                            'swvl4',
                            't2m',
                            'tp',
                            'u10',
                            'v10'
                        ],
                        help='input variables_era5_land')

    parser.add_argument('--variables_cpc', type=str,
                        default=['precip'],
                        help='input variables_cpc')

    parser.add_argument('--variables_hres_forecast', type=str,
                        default=[
                            'e',
                            'sf',
                            'sp',
                            'ssr',
                            'str',
                            't2m',
                            'tp'
                        ],
                        help='input variables_hres')

    parser.add_argument('--variables_static', type=str,
                        default=[
                            'CalChanMan1', 'CalChanMan2', 'GwLoss', 'GwPercValue', 'LZTC', 'LZthreshold',
                            'LakeMultiplier', 'PowerPrefFlow', 'QSplitMult', 'ReservoirRnormqMult', 'SnowMeltCoef',
                            'UZTC', 'adjustNormalFlood', 'b_Xinanjiang',
                            #'chan',
                            'chanbnkf', 'chanbw', 'chanflpn',
                            'changrad', 'chanlength', 'chanman',
                            #'chans',
                            'cropcoef', 'cropgrpn', 'elv', 'elvstd',
                            #'enconsuse',
                            'fracforest', 'fracgwused', 'fracirrigated', 'fracncused', 'fracother',
                            'fracrice', 'fracsealed', 'fracwater', 'genua1', 'genua2', 'genua3', 'gradient',
                            'gwbodies', 'ksat1', 'ksat2', 'ksat3', 'laif_01', 'laif_02', 'laif_03', 'laif_04',
                            'laif_05', 'laif_06', 'laif_07', 'laif_08', 'laif_09', 'laif_10', 'laif_11', 'laif_12',
                            'laii_01', 'laii_02', 'laii_03', 'laii_04', 'laii_05', 'laii_06', 'laii_07', 'laii_08',
                            'laii_09', 'laii_10', 'laii_11', 'laii_12', 'laio_01', 'laio_02', 'laio_03', 'laio_04',
                            'laio_05', 'laio_06', 'laio_07', 'laio_08', 'laio_09', 'laio_10', 'laio_11', 'laio_12',
                            #'lakea', 'lakearea', 'lakeavinflow',
                            'lambda1', 'lambda2', 'lambda3', 'ldd',
                            #'lusemask',
                            'mannings',
                            #'outlets',
                            'pixarea', 'pixleng',
                            #'rclim', 'rflim_97th',
                            'riceharvestday1', 'riceharvestday2', 'riceharvestday3', 'riceplantingday1', 'riceplantingday2',
                            'riceplantingday3',
                            #'rminq_5th', 'rndq_97th', 'rnlim_67th', 'rnormq_50th', 'rstor',
                            #'soildepth1',
                            'soildepth2', 'soildepth3',
                            #'thetar1', 'thetar2', 'thetar3',
                            'thetas1', 'thetas2', 'thetas3', 'upArea', 'waterregions'
                        ],
                       # default=[
                            #'aet_mm_c01', 'aet_mm_c02', 'aet_mm_c03', 'aet_mm_c04', 'aet_mm_c05', 'aet_mm_c06',
                            #'aet_mm_c07', 'aet_mm_c08', 'aet_mm_c09', 'aet_mm_c10', 'aet_mm_c11', 'aet_mm_c12',
                        #    'aet_mm_cyr', 'aet_mm_uyr', 'annualSnowFraction_fs',
                        #    'ari_ix_cav', 'ari_ix_uav',
                        #    'aridity_Im', 'cly_pc_cav', 'cly_pc_uav',
                            #'cmi_ix_c01', 'cmi_ix_c02', 'cmi_ix_c03', 'cmi_ix_c04', 'cmi_ix_c05', 'cmi_ix_c06',
                            #'cmi_ix_c07', 'cmi_ix_c08', 'cmi_ix_c09', 'cmi_ix_c10', 'cmi_ix_c11', 'cmi_ix_c12',
                        #    'cmi_ix_cyr', 'cmi_ix_uyr', 'crp_pc_cse', 'crp_pc_use',
                        #    'dis_m3_pmn', 'dis_m3_pmx', 'dis_m3_pyr', 'dor_pc_pva', 'ele_mt_cav',
                        #    'ele_mt_cmn', 'ele_mt_cmx', 'ele_mt_uav', 'ero_kh_cav', 'ero_kh_uav', 'for_pc_cse',
                        #    'for_pc_use', 'gdp_ud_cav', 'gdp_ud_csu', 'gdp_ud_usu',
                        #    'gla_pc_cse', 'gla_pc_use',
                            #'glc_pc_c01', 'glc_pc_c02', 'glc_pc_c03', 'glc_pc_c04', 'glc_pc_c05', 'glc_pc_c06',
                            #'glc_pc_c07', 'glc_pc_c08', 'glc_pc_c09', 'glc_pc_c10', 'glc_pc_c11', 'glc_pc_c12',
                            #'glc_pc_c13', 'glc_pc_c14', 'glc_pc_c15', 'glc_pc_c16', 'glc_pc_c17', 'glc_pc_c18',
                            #'glc_pc_c19', 'glc_pc_c20', 'glc_pc_c21', 'glc_pc_c22', 'glc_pc_u01', 'glc_pc_u02',
                            #'glc_pc_u03', 'glc_pc_u04', 'glc_pc_u05', 'glc_pc_u06', 'glc_pc_u07', 'glc_pc_u08',
                            #'glc_pc_u09', 'glc_pc_u10', 'glc_pc_u11', 'glc_pc_u12', 'glc_pc_u13', 'glc_pc_u14',
                            #'glc_pc_u15', 'glc_pc_u16', 'glc_pc_u17', 'glc_pc_u18', 'glc_pc_u19', 'glc_pc_u20',
                            #'glc_pc_u21', 'glc_pc_u22',
                        #    'glwd_delta_area_ha', 'glwd_delta_area_pct', 'glwd_delta_main_class_50pct', 'gwt_cm_cav',
                        #    'hdi_ix_cav', 'hft_ix_c09', 'hft_ix_c93', 'hft_ix_u09', 'hft_ix_u93',
                        #    'hyd_glo_ldn', 'hyd_glo_lup', 'inu_pc_clt', 'inu_pc_cmn',
                        #    'inu_pc_cmx', 'inu_pc_ult', 'inu_pc_umn', 'inu_pc_umx', 'ire_pc_cse', 'ire_pc_use',
                        #    'kar_pc_cse', 'kar_pc_use', 'ldd', 'lka_pc_cse', 'lka_pc_use',
                        #    'lkv_mc_usu', 'nli_ix_cav', 'nli_ix_uav', 'pac_pc_cse', 'pac_pc_use',
                            #'pet_mm_c01',
                            #'pet_mm_c02', 'pet_mm_c03', 'pet_mm_c04', 'pet_mm_c05', 'pet_mm_c06', 'pet_mm_c07',
                            #'pet_mm_c08', 'pet_mm_c09', 'pet_mm_c10', 'pet_mm_c11', 'pet_mm_c12',
                        #    'pet_mm_cyr', 'pet_mm_uyr',
                            #'pnv_pc_c01', 'pnv_pc_c02', 'pnv_pc_c03', 'pnv_pc_c04',
                            #'pnv_pc_c05', 'pnv_pc_c06', 'pnv_pc_c07', 'pnv_pc_c08', 'pnv_pc_c09', 'pnv_pc_c10',
                            #'pnv_pc_c11', 'pnv_pc_c12', 'pnv_pc_c13', 'pnv_pc_c14', 'pnv_pc_c15', 'pnv_pc_u01',
                            #'pnv_pc_u02', 'pnv_pc_u03', 'pnv_pc_u04', 'pnv_pc_u05', 'pnv_pc_u06', 'pnv_pc_u07',
                            #'pnv_pc_u08', 'pnv_pc_u09', 'pnv_pc_u10', 'pnv_pc_u11', 'pnv_pc_u12', 'pnv_pc_u13',
                            #'pnv_pc_u14', 'pnv_pc_u15',
                        #    'pop_ct_csu', 'pop_ct_usu', 'ppd_pk_cav', 'ppd_pk_uav',
                            #'pre_mm_c01', 'pre_mm_c02', 'pre_mm_c03', 'pre_mm_c04', 'pre_mm_c05', 'pre_mm_c06',
                            #'pre_mm_c07', 'pre_mm_c08', 'pre_mm_c09', 'pre_mm_c10', 'pre_mm_c11', 'pre_mm_c12',
                        #    'pre_mm_cyr', 'pre_mm_uyr', 'prm_pc_cse', 'prm_pc_use', 'pst_pc_cse', 'pst_pc_use',
                        #    'rdd_mk_cav', 'rdd_mk_uav', 'rev_mc_usu', 'ria_ha_csu', 'ria_ha_usu',
                        #    'riv_tc_csu', 'riv_tc_usu', 'run_mm_cyr', 'seasonalityOfAridity_Imr', 'sgr_dk_rav',
                        #    'slp_dg_cav', 'slp_dg_uav', 'slt_pc_cav', 'slt_pc_uav', 'snd_pc_cav', 'snd_pc_uav',
                            #'snw_pc_c01', 'snw_pc_c02', 'snw_pc_c03', 'snw_pc_c04', 'snw_pc_c05', 'snw_pc_c06',
                            #'snw_pc_c07', 'snw_pc_c08', 'snw_pc_c09', 'snw_pc_c10', 'snw_pc_c11', 'snw_pc_c12',
                        #    'snw_pc_cmx', 'snw_pc_cyr',
                        #    'snw_pc_uyr', 'soc_th_cav', 'soc_th_uav', 'stream_pow',
                            #'swc_pc_c01', 'swc_pc_c02', 'swc_pc_c03', 'swc_pc_c04', 'swc_pc_c05', 'swc_pc_c06',
                            #'swc_pc_c07', 'swc_pc_c08', 'swc_pc_c09', 'swc_pc_c10', 'swc_pc_c11', 'swc_pc_c12',
                        #    'swc_pc_cyr', 'swc_pc_uyr',
                            #'tmp_dc_c01', 'tmp_dc_c02',
                            #'tmp_dc_c03', 'tmp_dc_c04', 'tmp_dc_c05', 'tmp_dc_c06', 'tmp_dc_c07', 'tmp_dc_c08',
                            #'tmp_dc_c09', 'tmp_dc_c10', 'tmp_dc_c11', 'tmp_dc_c12',
                        #    'tmp_dc_cmn', 'tmp_dc_cmx',
                        #    'tmp_dc_cyr', 'tmp_dc_uyr', 'uparea', 'urb_pc_cse', 'urb_pc_use',
                            #'wet_pc_c01', 'wet_pc_c02', 'wet_pc_c03', 'wet_pc_c04', 'wet_pc_c05', 'wet_pc_c06',
                            #'wet_pc_c07', 'wet_pc_c08', 'wet_pc_c09', 'wet_pc_cg1', 'wet_pc_cg2', 'wet_pc_u01',
                            #'wet_pc_u02', 'wet_pc_u03', 'wet_pc_u04', 'wet_pc_u05', 'wet_pc_u06', 'wet_pc_u07',
                            #'wet_pc_u08', 'wet_pc_u09',
                        #    'wet_pc_ug1', 'wet_pc_ug2',

                       #     ],
                        help='input variables_static')

    # --- variables for log1p transformations --- #
    parser.add_argument('--variables_glofas_log1p', type=str,
                        default=None,
                        #default=[
                        #    'acc_rod24',
                        #    'dis24',
                        #    'sd',
                        #],
                        help='glofas variables that need log1p transformation')

    parser.add_argument('--variables_era5_land_log1p', type=str,
                        default=None,
                        help='era5 land variables that need log1p transformation')

    parser.add_argument('--variables_hres_forecast_log1p', type=str,
                        default=None,
                        help='hres forecast variables that need log1p transformation')

    parser.add_argument('--variables_cpc_log1p', type=str,
                        default=['precip'],
                        help='cpc variables that need log1p transformation')

    parser.add_argument('--variables_static_log1p', type=str,
                        #default=[#'ari_ix_cav', 'ari_ix_uav',
                        #         'dis_m3_pmn', 'dis_m3_pmx', 'dis_m3_pyr',
                        #         'dor_pc_pva', 'ero_kh_cav', 'ero_kh_uav', 'gdp_ud_cav', 'gdp_ud_csu',
                        #         'gdp_ud_usu'],
                                 #'hyd_glo_ldn', 'hyd_glo_lup', 'lkv_mc_usu', 'nli_ix_cav',
                                 #'nli_ix_uav', 'pop_ct_csu', 'pop_ct_usu', 'ppd_pk_cav', 'ppd_pk_uav',
                                 #'pre_mm_c01', 'pre_mm_c02', 'pre_mm_c03', 'pre_mm_c04', 'pre_mm_c05',
                                 #'pre_mm_c06', 'pre_mm_c07', 'pre_mm_c08', 'pre_mm_c09', 'pre_mm_c10',
                                 #'pre_mm_c11', 'pre_mm_c12', 'pre_mm_cyr', 'pre_mm_uyr', 'rdd_mk_cav',
                                 #'rdd_mk_uav', 'rev_mc_usu', 'rev_mc_usu', 'ria_ha_csu', 'ria_ha_usu',
                                 #'riv_tc_csu', 'riv_tc_usu', 'sgr_dk_rav', 'stream_pow', 'uparea'],
                        default=[
                            "chanbw", "chanflpn", "elvstd", "ksat1", "ksat2", "ksat3", "soildepth2", "soildepth3",
                            "upArea", "waterregions"
                        ],
                     #   default = ['dis_m3_pmn', 'dis_m3_pmx', 'dis_m3_pyr', 'ero_kh_cav', 'ero_kh_uav', 'gdp_ud_cav',
                     #         'gdp_ud_csu', 'gdp_ud_usu', 'hyd_glo_ldn', 'hyd_glo_lup', 'lkv_mc_usu', 'pop_ct_csu',
                     #         'pop_ct_usu', 'ppd_pk_cav', 'ppd_pk_uav', 'pre_mm_cyr', 'pre_mm_uyr', 'rdd_mk_cav',
                     #         'rdd_mk_uav', 'rev_mc_usu', 'rev_mc_usu', 'ria_ha_csu', 'ria_ha_usu', 'riv_tc_csu',
                     #         'riv_tc_usu', 'sgr_dk_rav', 'ari_ix_cav', 'ari_ix_uav', 'uparea', 'stream_pow',
                     #         'nli_ix_cav', 'nli_ix_uav', 'dor_pc_pva'],
                        help='static variables that need log1p transformation')

    return parser


def read_arguments(train=True, print=True, save=True):
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
            if k in ['variables_glofas', 'variables_era5_land', 'variables_static', 'variables_hres_forecast',
                     'variables_glofas_log1p', 'variables_era5_land_log1p', 'variables_static_log1p', 'variables_hres_forecast_log1p'
                     'years_train', 'years_val', 'years_test', 'dir_log', 'variables_cpc', 'variables_cpc_log1p',
                     'root_glofas_reanalysis', 'root_era5_land_reanalysis','root_cpc', 'root_static', 'root_hres_forecast',
                     'root_obs']:
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
        comment = '\t[default: %s]' % str(parser.get_default('root_hres_forecast'))
        message += '{:>25}: {:<20}{}\n'.format('root_hres_forecast',
                                                 vars(config)['root_hres_forecast'], comment)
        comment = '\t[default: %s]' % str(parser.get_default('root_cpc'))
        message += '{:>25}: {:<20}{}\n'.format('root_cpc', vars(config)['root_cpc'], comment)
        comment = '\t[default: %s]' % str(parser.get_default('root_static'))
        message += '{:>25}: {:<20}{}\n'.format('root_static', vars(config)['root_static'], comment)
        comment = '\t[default: %s]' % str(parser.get_default('root_obs'))
        message += '{:>25}: {:<20}{}\n'.format('root_obs', vars(config)['root_obs'], comment)

        comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
        message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

        message += '\n----------------- Input Variables -------      -------------------'
        message += '\n\n{:>20}: {}\n'.format('Variables GloFAS', str(config.variables_glofas))
        message += '{:>20}: {}\n'.format('Variables Era5-Land', str(config.variables_era5_land))
        message += '{:>20}: {}\n'.format('Variables hres forecast', str(config.variables_hres_forecast))#
        message += '{:>20}: {}\n'.format('Variables CPC', str(config.variables_cpc))
        message += '{:>20}: {}\n'.format('Variables Static', str(config.variables_static))

        message += '\n{:>26}: {}\n'.format('Variables GloFAS log1p', str(config.variables_glofas_log1p))
        message += '{:>26}: {}\n'.format('Variables Era5-Land logp1', str(config.variables_era5_land_log1p))
        message += '{:>26}: {}\n'.format('Variables hres forecast logp1', str(config.variables_hres_forecast_log1p))
        message += '{:>26}: {}\n'.format('Variables CPC logp1', str(config.variables_cpc_log1p))
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
        if k in ['variables_glofas', 'variables_era5_land', 'variables_static', 'variables_hres_forecast',
                 'variables_glofas_log1p', 'variables_era5_land_log1p', 'variables_static_log1p',
                 'variables_hres_forecast_log1p', 'variables_cpc', 'variables_cpc_log1p',
                 'years_train', 'years_val', 'years_test', 'dir_log',
                 'root_glofas_reanalysis', 'root_era5_land_reanalysis', 'root_cpc', 'root_static', 'root_hres_forecast',
                 'root_obs']:
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
    comment = '\t[default: %s]' % str(parser.get_default('root_hres_forecast'))
    message += '{:>25}: {:<20}{}\n'.format('root_hres_forecast', vars(config)['root_hres_forecast'], comment)
    comment = '\t[default: %s]' % str(parser.get_default('root_cpc'))
    message += '{:>25}: {:<20}{}\n'.format('root_cpc', vars(config)['root_cpc'], comment)
    comment = '\t[default: %s]' % str(parser.get_default('root_static'))
    message += '{:>25}: {:<20}{}\n'.format('root_static', vars(config)['root_static'], comment)
    comment = '\t[default: %s]' % str(parser.get_default('root_obs'))
    message += '{:>25}: {:<20}{}\n'.format('root_obs', vars(config)['root_obs'], comment)

    comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
    message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

    message += '\n----------------- Input Variables -------      -------------------'
    message += '\n\n{:>20}: {}\n'.format('Variables GloFAS', str(config.variables_glofas))
    message += '{:>20}: {}\n'.format('Variables Era5-Land', str(config.variables_era5_land))
    message += '{:>20}: {}\n'.format('Variables hres forecast', str(config.variables_hres_forecast))
    message += '{:>20}: {}\n'.format('Variables CPC', str(config.variables_cpc))
    message += '{:>20}: {}\n'.format('Variables Static', str(config.variables_static))

    message += '\n{:>26}: {}\n'.format('Variables GloFAS log1p', str(config.variables_glofas_log1p))
    message += '{:>26}: {}\n'.format('Variables Era5-Land logp1', str(config.variables_era5_land_log1p))
    message += '{:>26}: {}\n'.format('Variables hres forecast logp1', str(config.variables_hres_forecast_log1p))
    message += '{:>26}: {}\n'.format('Variables CPC logp1', str(config.variables_cpc_log1p))
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


