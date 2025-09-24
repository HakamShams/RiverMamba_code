# ------------------------------------------------------------------
# Simple script to visualize the Static data
# ------------------------------------------------------------------

import xarray as xr
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

np.set_printoptions(suppress=True)
xr.set_options(display_max_rows=40)
xr.set_options(display_width=1000)

# ------------------------------------------------------------------

root_static = r'/home/ssd4tb/shams/GloFAS_Static/'

# define the dataset for visualization HydroRIVERS or LISFLOOD

dataset = 'LISFLOOD'

# define the variables to be visualized
# HydroRIVERS
variables_hydrorivers = ['aet_mm_c01', 'aet_mm_c02', 'aet_mm_c03', 'aet_mm_c04', 'aet_mm_c05', 'aet_mm_c06',
                         'aet_mm_c07', 'aet_mm_c08', 'aet_mm_c09', 'aet_mm_c10', 'aet_mm_c11', 'aet_mm_c12',
                         'aet_mm_cyr', 'aet_mm_uyr', 'annualSnowFraction_fs', 'ari_ix_cav', 'ari_ix_uav',
                         'aridity_Im', 'class_geom', 'class_hydr', 'class_phys', 'cls_cl_cmj', 'cly_pc_cav',
                         'cly_pc_uav', 'clz_cl_cmj', 'cmi_ix_c01', 'cmi_ix_c02', 'cmi_ix_c03', 'cmi_ix_c04',
                         'cmi_ix_c05', 'cmi_ix_c06', 'cmi_ix_c07', 'cmi_ix_c08', 'cmi_ix_c09', 'cmi_ix_c10',
                         'cmi_ix_c11', 'cmi_ix_c12', 'cmi_ix_cyr', 'cmi_ix_uyr', 'crp_pc_cse', 'crp_pc_use',
                         'dis_m3_pmn', 'dis_m3_pmx', 'dis_m3_pyr', 'dor_pc_pva', 'ele_mt_cav', 'ele_mt_cmn',
                         'ele_mt_cmx', 'ele_mt_uav', 'elevation', 'ero_kh_cav', 'ero_kh_uav', 'fec_cl_cmj',
                         'fmh_cl_cmj', 'for_pc_cse', 'for_pc_use', 'gad_id_cmj', 'gdp_ud_cav', 'gdp_ud_csu',
                         'gdp_ud_usu', 'gla_pc_cse', 'gla_pc_use', 'glc_cl_cmj', 'glc_pc_c01', 'glc_pc_c02',
                         'glc_pc_c03', 'glc_pc_c04', 'glc_pc_c05', 'glc_pc_c06', 'glc_pc_c07', 'glc_pc_c08',
                         'glc_pc_c09', 'glc_pc_c10', 'glc_pc_c11', 'glc_pc_c12', 'glc_pc_c13', 'glc_pc_c14',
                         'glc_pc_c15', 'glc_pc_c16', 'glc_pc_c17', 'glc_pc_c18', 'glc_pc_c19', 'glc_pc_c20',
                         'glc_pc_c21', 'glc_pc_c22', 'glc_pc_u01', 'glc_pc_u02', 'glc_pc_u03', 'glc_pc_u04',
                         'glc_pc_u05', 'glc_pc_u06', 'glc_pc_u07', 'glc_pc_u08', 'glc_pc_u09', 'glc_pc_u10',
                         'glc_pc_u11', 'glc_pc_u12', 'glc_pc_u13', 'glc_pc_u14', 'glc_pc_u15', 'glc_pc_u16',
                         'glc_pc_u17', 'glc_pc_u18', 'glc_pc_u19', 'glc_pc_u20', 'glc_pc_u21', 'glc_pc_u22',
                         'glwd_delta_area_ha', 'glwd_delta_area_pct', 'glwd_delta_main_class',
                         'glwd_delta_main_class_50pct', 'gwt_cm_cav', 'hdi_ix_cav', 'hft_ix_c09', 'hft_ix_c93',
                         'hft_ix_u09', 'hft_ix_u93', 'hyd_glo_ldn', 'hyd_glo_lup', 'inu_pc_clt', 'inu_pc_cmn',
                         'inu_pc_cmx', 'inu_pc_ult', 'inu_pc_umn', 'inu_pc_umx', 'ire_pc_cse', 'ire_pc_use',
                         'kar_pc_cse', 'kar_pc_use', 'kmeans_30', 'ldd', 'lit_cl_cmj', 'lka_pc_cse', 'lka_pc_use',
                         'lkv_mc_usu', 'nli_ix_cav', 'nli_ix_uav', 'pac_pc_cse', 'pac_pc_use', 'pet_mm_c01',
                         'pet_mm_c02', 'pet_mm_c03', 'pet_mm_c04', 'pet_mm_c05', 'pet_mm_c06', 'pet_mm_c07',
                         'pet_mm_c08', 'pet_mm_c09', 'pet_mm_c10', 'pet_mm_c11', 'pet_mm_c12', 'pet_mm_cyr',
                         'pet_mm_uyr', 'pnv_cl_cmj', 'pnv_pc_c01', 'pnv_pc_c02', 'pnv_pc_c03', 'pnv_pc_c04',
                         'pnv_pc_c05', 'pnv_pc_c06', 'pnv_pc_c07', 'pnv_pc_c08', 'pnv_pc_c09', 'pnv_pc_c10',
                         'pnv_pc_c11', 'pnv_pc_c12', 'pnv_pc_c13', 'pnv_pc_c14', 'pnv_pc_c15', 'pnv_pc_u01',
                         'pnv_pc_u02', 'pnv_pc_u03', 'pnv_pc_u04', 'pnv_pc_u05', 'pnv_pc_u06', 'pnv_pc_u07',
                         'pnv_pc_u08', 'pnv_pc_u09', 'pnv_pc_u10', 'pnv_pc_u11', 'pnv_pc_u12', 'pnv_pc_u13',
                         'pnv_pc_u14', 'pnv_pc_u15', 'pop_ct_csu', 'pop_ct_usu', 'ppd_pk_cav', 'ppd_pk_uav',
                         'pre_mm_c01', 'pre_mm_c02', 'pre_mm_c03', 'pre_mm_c04', 'pre_mm_c05', 'pre_mm_c06',
                         'pre_mm_c07', 'pre_mm_c08', 'pre_mm_c09', 'pre_mm_c10', 'pre_mm_c11', 'pre_mm_c12',
                         'pre_mm_cyr', 'pre_mm_uyr', 'prm_pc_cse', 'prm_pc_use', 'pst_pc_cse', 'pst_pc_use',
                         'rdd_mk_cav', 'rdd_mk_uav', 'reach_type', 'rev_mc_usu', 'ria_ha_csu', 'ria_ha_usu',
                         'riv_tc_csu', 'riv_tc_usu', 'run_mm_cyr', 'seasonalityOfAridity_Imr', 'sgr_dk_rav',
                         'slp_dg_cav', 'slp_dg_uav', 'slt_pc_cav', 'slt_pc_uav', 'snd_pc_cav', 'snd_pc_uav',
                         'snw_pc_c01', 'snw_pc_c02', 'snw_pc_c03', 'snw_pc_c04', 'snw_pc_c05', 'snw_pc_c06',
                         'snw_pc_c07', 'snw_pc_c08', 'snw_pc_c09', 'snw_pc_c10', 'snw_pc_c11', 'snw_pc_c12',
                         'snw_pc_cmx', 'snw_pc_cyr', 'snw_pc_uyr', 'soc_th_cav', 'soc_th_uav', 'stream_pow',
                         'swc_pc_c01', 'swc_pc_c02', 'swc_pc_c03', 'swc_pc_c04', 'swc_pc_c05', 'swc_pc_c06',
                         'swc_pc_c07', 'swc_pc_c08', 'swc_pc_c09', 'swc_pc_c10', 'swc_pc_c11', 'swc_pc_c12',
                         'swc_pc_cyr', 'swc_pc_uyr', 'tbi_cl_cmj', 'tec_cl_cmj', 'tmp_dc_c01', 'tmp_dc_c02',
                         'tmp_dc_c03', 'tmp_dc_c04', 'tmp_dc_c05', 'tmp_dc_c06', 'tmp_dc_c07', 'tmp_dc_c08',
                         'tmp_dc_c09', 'tmp_dc_c10', 'tmp_dc_c11', 'tmp_dc_c12', 'tmp_dc_cmn', 'tmp_dc_cmx',
                         'tmp_dc_cyr', 'tmp_dc_uyr', 'uparea', 'urb_pc_cse', 'urb_pc_use', 'wet_cl_cmj',
                         'wet_pc_c01', 'wet_pc_c02', 'wet_pc_c03', 'wet_pc_c04', 'wet_pc_c05', 'wet_pc_c06',
                         'wet_pc_c07', 'wet_pc_c08', 'wet_pc_c09', 'wet_pc_cg1', 'wet_pc_cg2', 'wet_pc_u01',
                         'wet_pc_u02', 'wet_pc_u03', 'wet_pc_u04', 'wet_pc_u05', 'wet_pc_u06', 'wet_pc_u07',
                         'wet_pc_u08', 'wet_pc_u09', 'wet_pc_ug1', 'wet_pc_ug2'
                        ]

# LISFLOOD
variables_lisflood = ['CalChanMan1', 'CalChanMan2', 'GwLoss', 'GwPercValue', 'LZTC', 'LZthreshold',
                      'LakeMultiplier', 'PowerPrefFlow', 'QSplitMult', 'ReservoirRnormqMult',
                      'SnowMeltCoef', 'UZTC', 'adjustNormalFlood', 'b_Xinanjiang', 'chanbnkf',
                      'chanbw', 'chanflpn', 'changrad', 'chanlength', 'chanman', 'cropcoef',
                      'cropgrpn', 'elv', 'elvstd', 'fracforest', 'fracgwused', 'fracirrigated',
                      'fracncused', 'fracother', 'fracrice', 'fracsealed', 'fracwater', 'genua1', 'genua2',
                      'genua3', 'gradient', 'gwbodies', 'ksat1', 'ksat2', 'ksat3', 'laif_01', 'laif_02',
                      'laif_03', 'laif_04', 'laif_05', 'laif_06', 'laif_07', 'laif_08', 'laif_09', 'laif_10',
                      'laif_11', 'laif_12', 'laii_01', 'laii_02', 'laii_03', 'laii_04', 'laii_05', 'laii_06',
                      'laii_07', 'laii_08', 'laii_09', 'laii_10', 'laii_11', 'laii_12', 'laio_01', 'laio_02',
                      'laio_03', 'laio_04', 'laio_05', 'laio_06', 'laio_07', 'laio_08', 'laio_09', 'laio_10',
                      'laio_11', 'laio_12', 'lambda1', 'lambda2', 'lambda3',
                      'ldd', 'mannings', 'pixarea', 'pixleng',
                      'riceharvestday1', 'riceharvestday2', 'riceharvestday3', 'riceplantingday1', 'riceplantingday2',
                      'riceplantingday3', 'soildepth2', 'soildepth3', 'thetar1', 'thetar2', 'thetar3', 'thetas1',
                      'thetas2', 'thetas3', 'upArea', 'waterregions'
                      ]

# ------------------------------------------------------------------

# read the netcdf data
if dataset == 'LISFLOOD':
    data = xr.open_dataset(os.path.join(root_static, "NeuralFAS_LISFLOOD_static.nc"))
    variables = variables_lisflood

elif dataset == 'HydroRIVERS':
    data = xr.open_dataset(os.path.join(root_static, "NeuralFAS_HydroRIVERS_static.nc"))
    variables = variables_hydrorivers
else:
    raise ValueError('Dataset is not supported')

# visualize each variable separately
for v in variables:

    data_v = data[v].values

    plt.imshow(data_v)
    plt.title('variable=' + v + ', dataset=' + dataset)
    plt.colorbar()

    plt.show()
    plt.close()

