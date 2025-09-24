
import numpy as np
import xarray as xr
import os
import json
from torch.utils.data import Dataset
import warnings
from datetime import datetime, timedelta
import time
import matplotlib

np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
import dask

dask.config.set(scheduler='synchronous')

# ------------------------------------------------------------------

class GloFAS_Dataset_edlstm(Dataset):
    """
    PyTorch Dataset for loading GloFAS flood forecasting data with multiple sources.
    Supports hindcast, forecast, static, and observational data as well as weighted target importance.

    Input features include:
        - GloFAS reanalysis (hindcast)
        - ERA5-Land (hindcast + forecast)
        - HRES forecast (optional)
        - CPC precipitation (hindcast)
        - Static attributes (e.g., elevation, soil, LAI)

    Output:
        - Multi-step targets as log1p-transformed discharge increments
        - Optional observational targets
        - Flood threshold severity maps
        - Time-dependent weight maps for loss weighting
    """
    def __init__(self,
                 root_glofas_reanalysis: str,
                 root_era5_land_reanalysis: str,
                 root_hres_forecast: str, # add hres for fine-tuning
                 root_static: str,
                 root_obs: str,
                 root_cpc: str,
                 is_hres_forecast: bool = True,
                 nan_fill: float = 0., 
                 delta_t: int = 4, # used to define the time steps collected for the input 
                 delta_t_f: int = 15, # used to define the forecast time steps
                 is_random_t: bool = True,
                 is_aug: bool = False,
                 is_shuffle: bool = False,
                 variables_glofas: list = None,
                 variables_era5_land: list = None,
                 variables_static: list = None,
                 variables_hres_forecast: list = None,
                 variables_glofas_log1p: list = None,
                 variables_era5_land_log1p: list = None,
                 variables_static_log1p: list = None,
                 variables_hres_forecast_log1p: list = None,
                 is_norm: bool = True,
                 years: list = None,
                 lat_min: float = None,
                 lat_max: float = None,
                 lon_min: float = None,
                 lon_max: float = None,
                 is_target_obs: bool = True):

        super().__init__()

        print("Initializing GloFAS_Dataset_edlstm...")

        self.root_glofas_reanalysis = root_glofas_reanalysis
        self.root_era5_land_reanalysis = root_era5_land_reanalysis
        self.root_static = root_static
        self.root_hres_forecast = root_hres_forecast
        self.root_obs = root_obs
        self.root_cpc = root_cpc
        self.nan_fill = nan_fill
        self.delta_t = delta_t
        self.delta_t_f = delta_t_f
        self.is_hres_forecast = is_hres_forecast # if we use hres for fine-tuning
        self.is_random_t = is_random_t
        self.is_aug = is_aug
        self.is_shuffle = is_shuffle
        self.is_norm = is_norm
        self.years = years
        self.variables_glofas = variables_glofas
        self.variables_era5_land = variables_era5_land
        self.variables_static = variables_static
        self.variables_hres_forecast = variables_hres_forecast
        self.variables_glofas_log1p = variables_glofas_log1p
        self.variables_era5_land_log1p = variables_era5_land_log1p
        self.variables_static_log1p = variables_static_log1p
        self.variables_hres_forecast_log1p = variables_hres_forecast_log1p
        self.variables_cpc = ['precip']
        self.variables_cpc_log1p = None
        self.is_target_obs=is_target_obs

        if self.variables_glofas_log1p:
            self._variables_glofas_log1p_indices = [self.variables_glofas.index(v) for v in self.variables_glofas_log1p]
            self._variables_glofas_norm_indices = [x for x in range(len(self.variables_glofas)) if x not in self._variables_glofas_log1p_indices] # for what ???
        if self.variables_era5_land_log1p:
            self._variables_era5_land_log1p_indices = [self.variables_era5_land.index(v) for v in self.variables_era5_land_log1p]
        if self.variables_static_log1p:
            self._variables_static_log1p_indices = [self.variables_static.index(v) for v in self.variables_static_log1p]

        if self.variables_hres_forecast_log1p:
            self._variables_hres_forecast_log1p_indices = [self.variables_hres_forecast.index(v) for v in
                                                           self.variables_hres_forecast_log1p]

        if self.variables_cpc_log1p:
            self._variables_cpc_log1p_indices = [self.variables_cpc_log1p.index(v) for v in self.variables_cpc_log1p]

        self.random_indices = None

        self.alpha = 0.25

        weights_t = np.exp(np.abs(np.arange(1, self.delta_t_f + 1) - self.delta_t_f - 1) * self.alpha)

        self.weights_t = weights_t[None, :, None]  # shape: (1, 7, 1)

        if lat_min is None:
            self.lat_min = -90
        else:
            self.lat_min = lat_min
        if lat_max is None:
            self.lat_max = +90
        else:
            self.lat_max = lat_max
        if lon_min is None:
            self.lon_min = -180
        else:
            self.lon_min = lon_min
        if lon_max is None:
            self.lon_max = +180
        else:
            self.lon_max = lon_max

        # sort years
        self.years.sort()

        # preprocessing for the dataset
        self.__get_path()
        self.__get_var_n()
        self.__load_glofas_statistic()
        self.__load_era5_land_statistic()
        self.__load_static_statistic()
        self.__load_hres_statistic()
        self.__load_target_statistic()
        self.__load_cpc_statistic()
        self.__load_flood_thresholds()
        self.__load_static_data()

        if is_shuffle:
            np.random.shuffle(self.files)


    def __load_glofas_statistic(self):
        """
        Private method to get the statistics of the GloFAS reanalysis dataset from the root directory
        """

        with open(os.path.join(self.root_glofas_reanalysis, 'GloFAS_statistics_train.json'), 'r') as file:
            dict_tmp = json.load(file)

            self.glofas_min, self.glofas_max, self.glofas_mean, self.glofas_std = [], [], [], []

            for v in self.variables_glofas:
                if self.variables_glofas_log1p:
                    if v in self.variables_glofas_log1p:
                        self.glofas_min.append(float(dict_tmp['log1p']['min'][v]))
                        self.glofas_max.append(float(dict_tmp['log1p']['max'][v]))
                        self.glofas_mean.append(float(dict_tmp['log1p']['mean'][v]))
                        self.glofas_std.append(float(dict_tmp['log1p']['std'][v]))
                        continue
                    else:
                        pass

                self.glofas_min.append(float(dict_tmp['min'][v]))
                self.glofas_max.append(float(dict_tmp['max'][v]))
                self.glofas_mean.append(float(dict_tmp['mean'][v]))
                self.glofas_std.append(float(dict_tmp['std'][v]))

            self.glofas_min = np.array(self.glofas_min)
            self.glofas_max = np.array(self.glofas_max)
            self.glofas_mean = np.array(self.glofas_mean)
            self.glofas_std = np.array(self.glofas_std)

    def __load_target_statistic(self):
        """
        Private method to get the statistics of the GloFAS reanalysis dataset from the root directory
        """

        with open(os.path.join(self.root_glofas_reanalysis, 'GloFAS_statistics_train_increment_sep.json'), 'r') as file:
            dict_tmp = json.load(file)

            self.glofas_t_min, self.glofas_t_max, self.glofas_t_mean, self.glofas_t_std = [], [], [], []

            variables=['dis24_inc_01', 'dis24_inc_02', 'dis24_inc_03', 'dis24_inc_04',
                   'dis24_inc_05', 'dis24_inc_06', 'dis24_inc_07',
                   'dis24_inc_08', 'dis24_inc_09', 'dis24_inc_10', 'dis24_inc_11',
                   'dis24_inc_12', 'dis24_inc_13', 'dis24_inc_14', 'dis24_inc_15']

            for v in variables:
                self.glofas_t_min.append(float(dict_tmp['log1p']['min'][v]))
                self.glofas_t_max.append(float(dict_tmp['log1p']['max'][v]))
                self.glofas_t_mean.append(float(dict_tmp['log1p']['mean'][v]))
                self.glofas_t_std.append(float(dict_tmp['log1p']['std'][v]))

            self.glofas_t_min = np.array(self.glofas_t_min)
            self.glofas_t_max = np.array(self.glofas_t_max)
            self.glofas_t_mean = np.array(self.glofas_t_mean)
            self.glofas_t_std = np.array(self.glofas_t_std)

    def __load_era5_land_statistic(self):
        """
        Private method to get the statistics of the Era5 Land dataset from the root directory
        """

        with open(os.path.join(self.root_era5_land_reanalysis, 'ERA5_Land_statistics_train.json'), 'r') as file:
            dict_tmp = json.load(file)

            self.era5_land_min, self.era5_land_max, self.era5_land_mean, self.era5_land_std = [], [], [], []

            for v in self.variables_era5_land:
                if self.variables_era5_land_log1p:
                    if v in self.variables_era5_land_log1p:
                        self.era5_land_min.append(float(dict_tmp['log1p']['min'][v]))
                        self.era5_land_max.append(float(dict_tmp['log1p']['max'][v]))
                        self.era5_land_mean.append(float(dict_tmp['log1p']['mean'][v]))
                        self.era5_land_std.append(float(dict_tmp['log1p']['std'][v]))
                        continue
                    else:
                        pass

                self.era5_land_min.append(float(dict_tmp['min'][v]))
                self.era5_land_max.append(float(dict_tmp['max'][v]))
                self.era5_land_mean.append(float(dict_tmp['mean'][v]))
                self.era5_land_std.append(float(dict_tmp['std'][v]))

            self.era5_land_min = np.array(self.era5_land_min)
            self.era5_land_max = np.array(self.era5_land_max)
            self.era5_land_mean = np.array(self.era5_land_mean)
            self.era5_land_std = np.array(self.era5_land_std)

    def __load_hres_statistic(self):
        """
        Private method to get the statistics of the hres dataset from the root directory
        """

        if self.is_hres_forecast:
            root_statistic = os.path.join(self.root_hres_forecast, 'hres_statistics_train.json')
        else:
            root_statistic = os.path.join(self.root_era5_land_reanalysis, 'ERA5_Land_statistics_train.json')

        with open(root_statistic, 'r') as file:
            dict_tmp = json.load(file)

            self.hres_min, self.hres_max, self.hres_mean, self.hres_std = [], [], [], []

            for v in self.variables_hres_forecast:
                if self.variables_hres_forecast_log1p:
                    if v in self.variables_hres_forecast_log1p:
                        self.hres_min.append(float(dict_tmp['log1p']['min'][v]))
                        self.hres_max.append(float(dict_tmp['log1p']['max'][v]))
                        self.hres_mean.append(float(dict_tmp['log1p']['mean'][v]))
                        self.hres_std.append(float(dict_tmp['log1p']['std'][v]))
                        continue
                    else:
                        pass

                self.hres_min.append(float(dict_tmp['min'][v]))
                self.hres_max.append(float(dict_tmp['max'][v]))
                self.hres_mean.append(float(dict_tmp['mean'][v]))
                self.hres_std.append(float(dict_tmp['std'][v]))

            self.hres_min = np.array(self.hres_min)
            self.hres_max = np.array(self.hres_max)
            self.hres_mean = np.array(self.hres_mean)
            self.hres_std = np.array(self.hres_std)

    def __load_cpc_statistic(self):
        """
        Private method to get the statistics of the cpc dataset from the root directory
        """

        root_statistic = os.path.join(self.root_cpc, 'CPC_statistics_train.json')

        with open(root_statistic, 'r') as file:

            dict_tmp = json.load(file)

            self.cpc_min, self.cpc_max, self.cpc_mean, self.cpc_std = [], [], [], []

            for v in self.variables_cpc:
                if self.variables_cpc_log1p:
                    if v in self.variables_cpc_log1p:
                        self.cpc_min.append(float(dict_tmp['log1p']['min'][v]))
                        self.cpc_max.append(float(dict_tmp['log1p']['max'][v]))
                        self.cpc_mean.append(float(dict_tmp['log1p']['mean'][v]))
                        self.cpc_std.append(float(dict_tmp['log1p']['std'][v]))
                        continue
                    else:
                        pass

                self.cpc_min.append(float(dict_tmp['min'][v]))
                self.cpc_max.append(float(dict_tmp['max'][v]))
                self.cpc_mean.append(float(dict_tmp['mean'][v]))
                self.cpc_std.append(float(dict_tmp['std'][v]))

            self.cpc_min = np.array(self.cpc_min)
            self.cpc_max = np.array(self.cpc_max)
            self.cpc_mean = np.array(self.cpc_mean)
            self.cpc_std = np.array(self.cpc_std)

    def __load_static_statistic(self):
        """
        Private method to get the statistics of the Static dataset from the root directory
        """

        with open(os.path.join(self.root_static, 'Static_LISFLOOD_statistics.json'), 'r') as file:
            dict_tmp = json.load(file)

            self.static_min, self.static_max, self.static_mean, self.static_std = [], [], [], []

            for v in self.variables_static:
                if self.variables_static_log1p:
                    if v in self.variables_static_log1p:
                        self.static_min.append(float(dict_tmp['log1p']['min'][v]))
                        self.static_max.append(float(dict_tmp['log1p']['max'][v]))
                        self.static_mean.append(float(dict_tmp['log1p']['mean'][v]))
                        self.static_std.append(float(dict_tmp['log1p']['std'][v]))
                        continue
                    else:
                        pass

                self.static_min.append(float(dict_tmp['min'][v]))
                self.static_max.append(float(dict_tmp['max'][v]))
                self.static_mean.append(float(dict_tmp['mean'][v]))
                self.static_std.append(float(dict_tmp['std'][v]))

            self.static_min = np.array(self.static_min)
            self.static_max = np.array(self.static_max)
            self.static_mean = np.array(self.static_mean)
            self.static_std = np.array(self.static_std)

    def __get_var_n(self):
        """
        Private method to get the number of variables
        """
        self.var_n_glofas = len(self.variables_glofas)
        self.var_n_era5_land = len(self.variables_era5_land)
        self.var_n_static = len(self.variables_static)
        self.var_n_hres = len(self.variables_hres_forecast)
        self.var_n_cpc = len(self.variables_cpc)

        self.var_n = self.var_n_glofas + self.var_n_era5_land + self.var_n_static + self.var_n_hres + self.var_n_cpc

    def __get_path(self):
        """
        Private method to get the dataset files inside the root directory
        self.files: list of dictionaries containing the file paths;
        self.files[x]: dictionary containing the file paths for each time window
        {
        'glofas': [file_day1.nc, file_day2.nc, file_day3.nc, file_day4.nc], # the last file in the hindcast sequence is the main file
        'era5': [era5_day1.nc, era5_day2.nc, era5_day3.nc, era5_day4.nc],
        'glofas_target': [file_day5.nc, file_day6.nc, …, file_day19.nc],
        'era5_forecast': [era5_day5.nc, …, era5_day19.nc],
        'hres_forecast': [file_hres.nc],   
        'cpc': [cpc_day1.nc, cpc_day2.nc, cpc_day3.nc, cpc_day4.nc],
        'obs_target': [obs_day5.nc, …, obs_day19.nc]
        }
        """

        self.files, self.data_time = [], []

        for year in self.years:

            year_dir_glofas = os.path.join(self.root_glofas_reanalysis, year)
            if not os.path.isdir(year_dir_glofas):
                raise ValueError('year {} does not exist in the GloFAS data'.format(year))

            year_dir_era5_land = os.path.join(self.root_era5_land_reanalysis, year)
            if not os.path.isdir(year_dir_era5_land):
                raise ValueError('year {} does not exist in the ERA5-Land data'.format(year))

            year_dir_hres = os.path.join(self.root_hres_forecast, year)
            if self.is_hres_forecast:
                if not os.path.isdir(year_dir_hres):
                    warnings.warn(f'Year {year} does not exist in the HRES data. Using ERA5 forecast instead.')

            files = os.listdir(year_dir_glofas)
            files = [file for file in files if file.endswith('.nc')]
            files.sort()

            for file in files: # the 'file' is the major time step for processing the current sample
                day, month = file[6:8], file[4:6]
                files_glofas, files_era5_land, files_time, files_obs, files_cpc = [], [], [], [], []
                file_name = year + month + day

                for delta in reversed(range(self.delta_t)):
                    # shift it for one day to make it easier for operational task
                    file_delta = (datetime(int(year), int(month), int(day)) - timedelta(days=delta+1)).strftime('%Y%m%d')
                    # here allows to read the data from the previous days from the previous year which is not in the same folder as the current year

                    file_glofas_delta = os.path.join(self.root_glofas_reanalysis, file_delta[:4], file_delta + '.nc')
                    if os.path.isfile(file_glofas_delta):
                        files_glofas.append(file_glofas_delta)
                    else:
                        warnings.warn('file {} does not exist in the GloFAS data'.format(file_glofas_delta))

                    files_time.append(self.get_day_of_year(file_delta))

                    file_delta_era5 = (datetime(int(year), int(month), int(day)) - timedelta(days=delta + 1)).strftime('%Y%m%d')

                    file_era5_land_delta = os.path.join(self.root_era5_land_reanalysis, file_delta_era5[:4],
                                                        file_delta_era5 + '.nc')

                    if os.path.isfile(file_era5_land_delta):
                        files_era5_land.append(file_era5_land_delta)
                    else:
                        warnings.warn('file {} does not exist in the ERA5-Land data'.format(file_era5_land_delta))

                    file_delta_cpc = (datetime(int(year), int(month), int(day)) - timedelta(days=delta + 2)).strftime('%Y%m%d')

                    file_cpc_delta = os.path.join(self.root_cpc, file_delta_cpc[:4], file_delta_cpc + '.nc')

                    if os.path.isfile(file_cpc_delta):
                        files_cpc.append(file_cpc_delta)
                    else:
                        warnings.warn('file {} does not exist in the CPC data'.format(file_cpc_delta))

                for delta in range(self.delta_t_f): ### question here for hres ###

                    delta += 1

                    file_delta = (datetime(int(year), int(month), int(day)) + timedelta(days=delta)).strftime('%Y%m%d')

                    file_glofas_delta = os.path.join(self.root_glofas_reanalysis, file_delta[:4], file_delta + '.nc')
                    if os.path.isfile(file_glofas_delta):
                        files_glofas.append(file_glofas_delta)
                    else:
                        warnings.warn('file {} does not exist in the GloFAS data'.format(file_glofas_delta))

                    file_era5_land_delta = os.path.join(self.root_era5_land_reanalysis, file_delta[:4],
                                                        file_delta + '.nc')
                    if os.path.isfile(file_era5_land_delta):
                        files_era5_land.append(file_era5_land_delta)
                    else:
                        warnings.warn('file {} does not exist in the ERA5-Land data'.format(file_era5_land_delta))

                    file_obs_delta = os.path.join(self.root_obs, file_delta[:4],
                                                        file_delta + '.nc')

                    if os.path.isfile(file_obs_delta):
                        files_obs.append(file_obs_delta)
                    else:
                        warnings.warn('file {} does not exist in the observation data'.format(file_obs_delta))

                    if self.is_hres_forecast and delta == 1:  # because hres has all time steps in the last day
                        file_hres = os.path.join(self.root_hres_forecast, file[:4], file)
                        if os.path.isfile(file_hres):
                            forecast_type = 'hres'  # Mark that we are using HRES
                        else:
                            warnings.warn(f'HRES file missing for {year}. Using ERA5 forecast instead.')
                            file_hres = None  # This will be replaced with ERA5 later
                            forecast_type = 'era5'  # Mark that we are using ERA5 instead

                    files_time.append(self.get_day_of_year(file_delta))
                if self.is_target_obs == True:
                    if len(files_glofas) == len(files_era5_land) == (self.delta_t + self.delta_t_f) and len(files_obs) == self.delta_t_f and len(files_cpc) == self.delta_t:
                        if self.is_hres_forecast:
                            self.files.append({'glofas': files_glofas[:self.delta_t],
                                            'era5': files_era5_land[:self.delta_t],
                                            'glofas_target': files_glofas[self.delta_t:],
                                            'era5_forecast': files_era5_land[self.delta_t:],
                                            'hres_forecast': [file_hres] if file_hres else [],
                                            'cpc': files_cpc,
                                            'obs_target': files_obs,
                                            'forecast_type': forecast_type,  # Store which forecast we are using
                                            'file_name': file_name
                                            })
                        else:
                            self.files.append({'glofas': files_glofas[:self.delta_t],
                                            'era5': files_era5_land[:self.delta_t],
                                            'glofas_target': files_glofas[self.delta_t:],
                                            'era5_forecast': files_era5_land[self.delta_t:],
                                            'cpc': files_cpc,
                                            'obs_target': files_obs,
                                            'file_name': file_name
                                            })

                        self.data_time.append(files_time[:self.delta_t])

                else:
                    if len(files_glofas) == len(files_era5_land) == (self.delta_t + self.delta_t_f) and len(files_cpc) == self.delta_t:
                        if self.is_hres_forecast:
                            self.files.append({'glofas': files_glofas[:self.delta_t],
                                            'era5': files_era5_land[:self.delta_t],
                                            'glofas_target': files_glofas[self.delta_t:],
                                            'era5_forecast': files_era5_land[self.delta_t:],
                                            'hres_forecast': [file_hres] if file_hres else [],
                                            'cpc': files_cpc,
                                            'obs_target': files_obs,
                                            'forecast_type': forecast_type,  # Store which forecast we are using
                                            'file_name': file_name
                                            })
                        else:
                            self.files.append({'glofas': files_glofas[:self.delta_t],
                                            'era5': files_era5_land[:self.delta_t],
                                            'glofas_target': files_glofas[self.delta_t:],
                                            'era5_forecast': files_era5_land[self.delta_t:],
                                            'cpc': files_cpc,
                                            'obs_target': files_obs,
                                            'file_name': file_name
                                            })

                        self.data_time.append(files_time[:self.delta_t])

        self.data_time = np.array(self.data_time).astype(np.int16)

        if len(self.files) == 0:
            raise ValueError('No files were found in the root directories')


    def __load_flood_thresholds(self): # change this to load the obs only flood threshold
        """
        Private method to get the flood threshold maps (9 severity levels) from the root static directory.
        Flood thresholds are computed for selected return periods i.e., of 1.5, 2, 5, 10, 20, 50, 100, 200, and 500 years.
        """


        if self.is_target_obs:
            files_thr = ['flood_threshold_obs_rl_1.5', 'flood_threshold_obs_rl_2.0',
                     'flood_threshold_obs_rl_5.0', 'flood_threshold_obs_rl_10.0',
                     'flood_threshold_obs_rl_20.0', 'flood_threshold_obs_rl_50.0',
                     'flood_threshold_obs_rl_100.0', 'flood_threshold_obs_rl_200.0',
                     'flood_threshold_obs_rl_500.0',]
            files_thr = [xr.open_dataset(os.path.join(self.root_static, "threshold_obs/grdc/" + file + ".nc")) for file in files_thr]
            self._thr = [1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        else:
            files_thr = ['flood_threshold_glofas_v4_rl_1.5', 'flood_threshold_glofas_v4_rl_2.0',
                     'flood_threshold_glofas_v4_rl_5.0', 'flood_threshold_glofas_v4_rl_10.0',
                     'flood_threshold_glofas_v4_rl_20.0', 'flood_threshold_glofas_v4_rl_50.0',
                     'flood_threshold_glofas_v4_rl_100.0', 'flood_threshold_glofas_v4_rl_200.0',
                     'flood_threshold_glofas_v4_rl_500.0',]
            files_thr = [xr.open_dataset(os.path.join(self.root_static, "threshold/" + file + ".nc")) for file in files_thr]
            self._thr = [1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

        self._flood_thresholds = xr.merge(files_thr)
        self._flood_thresholds = self._flood_thresholds.to_array().values


    def __load_static_data(self):

        with xr.open_dataset(os.path.join(self.root_static, "NeuralFAS_LISFLOOD_static.nc")) as dataset:

            self.data_static = dataset[self.variables_static]

            self.data_static = self.data_static.to_array().values

            self.n_stations = self.data_static.shape[1]

            if self.variables_static_log1p:
                self.data_static[self._variables_static_log1p_indices] = self.log1p_transform(
                    self.data_static[self._variables_static_log1p_indices])

            if self.is_norm:
                for v in range(len(self.static_mean)):

                    # replace the nan with min
                    self.data_static[v][np.isnan(self.data_static[v])] = self.static_min[v]
                    self.data_static[v] = self.min_max_scale(self.data_static[v],
                                                             self.static_min[v],
                                                             self.static_max[v],
                                                             -1,
                                                             1)

            self.data_static[np.isnan(self.data_static)] = self.nan_fill

        self.data_static = self.data_static.astype(np.float32)


    def min_max_scale(self, array: np.array,
                      min_alt: float, max_alt: float,
                      min_new: float = 0., max_new: float = 1.):

        """
        Helper method to normalize an array between new minimum and maximum values

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

        array = ((max_new - min_new) * (array - min_alt) / (max_alt - min_alt)) + min_new

        return array

    def get_day_of_year(self, file: str):
        """
        Helper method to get the year, month, day and week number from the file name

        Parameters
        ----------
        file : str
          name of the file in the dataset

        Returns
        ----------
        year : int
            corresponding year for the file
        month : int
            corresponding month number for the file
        day : int
            corresponding day of the month for the file
        week : int
            corresponding week number of the file
        """

        """
        https://stackoverflow.com/questions/620305/convert-year-month-day-to-day-of-year-in-python 
        given year, month, day return day of year Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 """

        file_name = os.path.splitext(os.path.basename(os.path.normpath(file)))[0]

        #year = int(file_name[:4])
        month = int(file_name[4:6])
        day = int(file_name[6:])

        return int((275 * month) / 9.0) - 1 * int((month + 9) / 12.0) + day - 30

    def __load_dynamic_data(self, NetCDF_files, dynamic_variables):
        """
        Private method to load NETCDF data from the files
        """

        datacube = xr.open_mfdataset(NetCDF_files,
                                     combine='nested',
                                     concat_dim='None',
                                     parallel=False,
                                     engine='netcdf4',
                                     )[dynamic_variables].to_array().values

        return datacube

    def __load_era5_data(self, NetCDF_files, dynamic_variables):
        """
        Private method to load NETCDF data from the files
        """

        datacube = xr.open_mfdataset(NetCDF_files,
                                     combine='nested',
                                     concat_dim='None',
                                     parallel=False,
                                     engine='netcdf4',
                                     )[dynamic_variables].to_array().values

        return datacube
    
    # add data loading for hres forecast
    def __load_hres_data(self, NetCDF_files, dynamic_variables, lead_time):
        """
        Private method to load NETCDF data from the files
        """

        if self.is_hres_forecast:
          
            datacube = xr.open_dataset(NetCDF_files[0]).isel(step=lead_time)[dynamic_variables].to_array().values
        else:
            if isinstance(lead_time, int):
                NetCDF_files = NetCDF_files[lead_time]
                datacube = xr.open_dataset(NetCDF_files)[dynamic_variables].to_array().values
            else:
                datacube = xr.open_mfdataset(NetCDF_files,
                                            combine='nested',
                                            concat_dim='None',
                                            parallel=False,
                                            engine='netcdf4',
                                            )[dynamic_variables].to_array().values
        return datacube

    # add data loading for cpc precipitation data
    def __load_cpc_data(self, NetCDF_files, dynamic_variables):
        """
        Private method to load NETCDF data from the files
        """

        datacube = xr.open_mfdataset(NetCDF_files,
                                     combine='nested',
                                     concat_dim='None',
                                     parallel=False,
                                     engine='netcdf4',
                                     )[dynamic_variables].to_array().values

        return datacube


    def log1p_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def log1p_inv_transform(self, x):
        return np.sign(x) * np.expm1(np.abs(x))

    def transform(self, x: np.ndarray, mean, std) -> np.ndarray:
        x_norm = (x - mean) / (std + 1e-6)
        return x_norm

    def inv_transform(self, x_norm: np.ndarray, mean, std) -> np.ndarray:
        x = (x_norm * (std + 1e-6)) + mean
        return x

    def get_flood_thresholds(self, thr: list = None): # do not need this function, just index self._flood_thresholds

        if thr is None:
            thr = self._thr
        assert set(thr).issubset(self._thr), "possible thresholds are 1.5, 2, 5, 10, 20, 50, 100, 200, 500"
        return self._flood_thresholds[[self._thr.index(item) for item in thr]] # get the threshold directly in the evaluation

    def generate_thr_weights(self, target_glofas, thresholds):

        """
        Generate severity-based weight map based on thresholds and target values.
        Also includes time weighting to emphasize recent forecast steps.
        """

        # target_glofas  1, 7, P
        # threshold P, 9
        # make sure target_glofas is (1, 7, P, 1)
        target_glofas = target_glofas[:, :, :, None]  # shape: (1, 7, P, 1)

        # make sure thresholds is (1, 1, P, 9)
        thresholds = thresholds.T[None, None, :, :]   # shape: (1, 1, P, 9)

        # broadcast comparison → (1, 7, P, 9)
        mask = (target_glofas >= thresholds)

        # self._thr: shape (9,)
        # broadcast to (1, 1, 1, 9) → then multiply with mask
        weights = mask * np.array(self._thr)[None, None, None, :]

        # max across 9 thresholds → shape: (1, 7, P)
        weight_map_thr = np.max(weights, axis=-1)

        # fallback weight if no threshold is triggered
        weight_map_thr[weight_map_thr == 0.] = 1

        # time weighting
        weight_map_thr = weight_map_thr * self.weights_t  # shape: (1, 7, P)

        # final transpose → (P, 7, 1)
        return weight_map_thr.transpose(2, 1, 0)

    def generate_thr_weights_obs(self, target_glofas, thresholds_obs):

        """
        Same as generate_thr_weights, but applies to observation-based thresholds.
        """

        # make sure target_glofas is (1, 7, P, 1)
        target_glofas = target_glofas[:, :, :, None]  # shape: (1, 7, P, 1)

        # make sure thresholds is (1, 1, P, 9)
        thresholds = thresholds_obs.T[None, None, :, :]   # shape: (1, 1, P, 9)

        # broadcast comparison → (1, 7, P, 9)
        mask = (target_glofas >= thresholds)

        # self._thr: shape (9,)
        # broadcast to (1, 1, 1, 9) → then multiply with mask
        weights = mask * np.array(self._thr)[None, None, None, :]

        # max across 9 thresholds → shape: (1, 7, P)
        weight_map_thr = np.max(weights, axis=-1)

        # fallback weight if no threshold is triggered
        weight_map_thr[weight_map_thr == 0.] = 1

        # time weighting
        weight_map_thr = weight_map_thr * self.weights_t  # shape: (1, 7, P)

        # final transpose → (P, 7, 1)
        return weight_map_thr.transpose(2, 1, 0)

    def __getitem__(self, index):

        """
        Load one training/validation sample consisting of:
            - hindcast inputs from GloFAS, ERA5, CPC
            - forecast inputs from HRES or ERA5
            - static data
            - log1p-transformed targets
            - optional observational targets
            - flood thresholds
            - lead time steps
            - dynamic weight map for loss weighting
        Output tensors are transposed to (Points, Seq, Features) format.
        """

        thresholds = self._flood_thresholds  # always loaded during __init__
        thresholds_obs = self._flood_thresholds  # same name, content differs based on is_target_obs

        # get the files in an updated way

        files_glofas, files_era5_land, files_glofas_t, files_cpc, files_obs_t = (self.files[index]['glofas'],
                                                                                 self.files[index]['era5'],
                                                                                 self.files[index]['glofas_target'],
                                                                                 self.files[index]['cpc'],
                                                                                 self.files[index]['obs_target']
                                                                                 )
        # get glofas variables
        data_glofas = self.__load_dynamic_data(files_glofas, self.variables_glofas)
        # get era5 land variables
        data_era5_land = self.__load_era5_data(files_era5_land, self.variables_era5_land)#.copy()
        # get cpc variables
        data_cpc = self.__load_cpc_data(files_cpc, self.variables_cpc)

        # get thresholds
        data_thresholds = self.get_flood_thresholds()#.copy()

        # get forecast variables -- choose either hres or era5
        files_hres = self.files[index]['hres_forecast']
        files_era5_forecast = self.files[index]['era5_forecast']
        forecast_type = self.files[index]['forecast_type']  # Get forecast type

        # get tagret variables for decoder
 
        target_glofas = self.__load_dynamic_data(files_glofas_t,
                                                    ['dis24'])
        lead_time = np.arange(1, self.delta_t_f + 1) # original shape in (15,)

        # data_hres V, T, P
        # Load HRES or ERA5 forecast data based on forecast_type
        if forecast_type == 'hres':
            data_hres = self.__load_hres_data(files_hres, self.variables_hres_forecast, lead_time - 1)#.copy()
        elif forecast_type == 'era5': # use the same variables required for hres but from era5
            data_hres = self.__load_era5_data(files_era5_forecast, self.variables_hres_forecast)#.copy()  # Use correct function

        # get obs data 1, T, P
        if self.is_target_obs:
            target_obs = self.__load_dynamic_data(files_obs_t,
                                                ['dis24'])
        else:
            target_obs = [None]  # Placeholder if not using target_obs

        if self.is_target_obs:
            weight_map_thr = self.generate_thr_weights_obs(target_obs, thresholds_obs)
        else:
            weight_map_thr = self.generate_thr_weights(target_glofas, thresholds)

        # update the target_glofas -- only calculate the increment to the last time step
        target_glofas = target_glofas - data_glofas[1:1 + 1, -1:, :]

        target_glofas = self.log1p_transform(target_glofas)

        if self.is_target_obs:
            target_obs = target_obs - data_glofas[1:1 + 1, -1:, :]
            target_obs = self.log1p_transform(target_obs)
        else:
            target_obs = None

        # log1p transformation -- apply for cpc, glofas -- also the static, but not done here
        if self.variables_glofas_log1p:
            data_glofas[self._variables_glofas_log1p_indices] = self.log1p_transform(data_glofas[self._variables_glofas_log1p_indices])
        if self.variables_era5_land_log1p:
            data_era5_land[self._variables_era5_land_log1p_indices] = self.log1p_transform(data_era5_land[self._variables_era5_land_log1p_indices])
        if self.variables_hres_forecast_log1p:
            data_hres[self._variables_hres_forecast_log1p_indices] = self.log1p_transform(
                data_hres[self._variables_hres_forecast_log1p_indices])
        data_cpc = self.log1p_transform(data_cpc)

        # normalization: apply for all the variables -- also the static, but not done here
        if self.is_norm:
            for v in range(self.var_n_glofas):
                data_glofas[v] = self.transform(data_glofas[v], self.glofas_mean[v], self.glofas_std[v])

            for v in range(self.var_n_era5_land):
                data_era5_land[v] = self.transform(data_era5_land[v], self.era5_land_mean[v], self.era5_land_std[v])
            for v in range(self.var_n_hres):
                data_hres[v] = self.transform(data_hres[v], self.hres_mean[v], self.hres_std[v])
            data_cpc = self.transform(data_cpc,self.cpc_mean,self.cpc_std)

        #  fill in the missing data
        data_era5_land[np.isnan(data_era5_land)] = self.nan_fill
        data_hres[np.isnan(data_hres)] = self.nan_fill

        if self.is_target_obs:

            return (data_glofas.transpose(2, 1, 0), data_era5_land.transpose(2, 1, 0), data_hres.transpose(2, 1, 0),self.data_static.transpose(1, 0), data_cpc.transpose(2, 1, 0),target_obs.transpose(2, 1, 0),
                data_thresholds.transpose(1, 0), target_glofas.transpose(2, 1, 0), np.array(lead_time),self.files[index]['file_name'],weight_map_thr) 

        else:
            # Create dummy target_obs with correct shape (e.g., (T, H, W))
            dummy_shape = target_glofas.transpose(2, 1, 0).shape  # same shape as target_glofas
            dummy_target_obs = np.full(dummy_shape, np.nan, dtype=np.float32)  # or use zeros
            return (
                data_glofas.transpose(2, 1, 0),
                data_era5_land.transpose(2, 1, 0),
                data_hres.transpose(2, 1, 0),
                self.data_static.transpose(1, 0),
                data_cpc.transpose(2, 1, 0),
                dummy_target_obs,
                data_thresholds.transpose(1, 0),
                target_glofas.transpose(2, 1, 0),
                np.array(lead_time),
                self.files[index]['file_name'],
                weight_map_thr
            )
    
    
    def __len__(self): # change
        """
        Method to get the number of files in the dataset
        """
        return len(self.files)
    

if __name__ == '__main__':
    # updates in the root path of training dataset
    root_glofas_reanalysis = r'/p/scratch/cesmtst/zhang36/NeuralFAS_dataset/GRDC_masked_dataset/GloFAS_Reanalysis'
    root_era5_land_reanalysis = r'/p/scratch/cesmtst/zhang36/NeuralFAS_dataset/GRDC_masked_dataset/ERA5-Land_Reanalysis_Global' 
    root_static = r'/p/scratch/cesmtst/zhang36/NeuralFAS_dataset/GRDC_masked_dataset/GloFAS_Static'
    root_hres_forecast = r'/p/scratch/cesmtst/zhang36/NeuralFAS_dataset/GRDC_masked_dataset/ECMWF_HRES_global'
    root_cpc = r'/p/scratch/cesmtst/zhang36/NeuralFAS_dataset/GRDC_masked_dataset/CPC_regridded_nearest'
    root_obs = r'/p/scratch/cesmtst/zhang36/NeuralFAS_dataset/GRDC_masked_dataset/GRDC_Obs'
    variables_glofas = ['acc_rod24', 'dis24', 'sd', 'swi']

    variables_era5_land = ['d2m', 'e', 'es', 'evabs', 'evaow', 'evatc', 'evavt', 'lai_hv', 'lai_lv', 'pev', 'sf',
                           'skt', 'slhf', 'smlt', 'sp', 'src', 'sro', 'sshf', 'ssr', 'ssrd', 'ssro',
                           'stl1', 'stl2', 'stl3', 'stl4', 'str', 'strd', 'swvl1', 'swvl2', 'swvl3', 'swvl4',
                           't2m', 'tp', 'u10', 'v10'
                           ]

    variables_glofas_log1p = None#['acc_rod24', 'dis24', 'sd']
    
    variables_static = ['CalChanMan1', 'CalChanMan2', 'GwLoss', 'GwPercValue', 'LZTC', 'LZthreshold',
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
                        'thetas2', 'thetas3', 'upArea', 'waterregions']
    
    variables_hres_forecast = ['e', 'sf', 'sp', 'ssr', 'str', 't2m', 'tp']

    variables_static_log1p = ["chanbw", "chanflpn", "elvstd", "ksat1", "ksat2", "ksat3", "soildepth2", "soildepth3",
                              "upArea", "waterregions"]

    years_train = [str(year) for year in range(2022, 2023)] # test only one year

    dataset = GloFAS_Dataset_edlstm(
        root_glofas_reanalysis=root_glofas_reanalysis,
        root_era5_land_reanalysis=root_era5_land_reanalysis,
        root_hres_forecast=root_hres_forecast,
        root_static=root_static,
        root_obs=root_obs,
        root_cpc=root_cpc,
        is_hres_forecast=True,
        nan_fill=0.,
        delta_t=5, # the length of the input -- it generates a continuous time series
        delta_t_f=7, # forecast lead time in total -- maximum 7 days????
        is_random_t=False,
        is_aug=False,
        is_shuffle=False,
        variables_glofas=variables_glofas,
        variables_era5_land=variables_era5_land,
        variables_static=variables_static,
        variables_hres_forecast=variables_hres_forecast,
        variables_glofas_log1p=variables_glofas_log1p,
        variables_era5_land_log1p=None,
        variables_static_log1p=variables_static_log1p,
        variables_hres_forecast_log1p=None,
        is_norm=True,
        years=years_train,
        lat_min=30,
        lat_max=60,
        lon_min=-10,
        lon_max=40
        )

    print('number of sampled data:', dataset.__len__()) # number of files in the dataset
    end = time.time()
    data_temp = dataset.__getitem__(0) 
    print('time: ', time.time() - end)

    print('data_glofas     shape: ', data_temp[0].shape) # (points, seq, number of feature)
    print('data_era5_land  shape: ', data_temp[1].shape) # (points, seq, number of feature)
    print('data_hres       shape: ', data_temp[2].shape) # (points, seq, number of feature)
    print('data_static     shape: ', data_temp[3].shape) # (Points, features)
    print('data_cpc        shape: ', data_temp[4].shape) # (points, seq, number of feature)
    print('target_obs      shape: ', data_temp[5].shape) # (points, forecast seq, target feature)
    print('data_thresholds shape: ', data_temp[6].shape) # 
    print('target_glofas   shape: ', data_temp[7].shape) # (points, forecast seq, target feature)
    print('data_lead_time  shape: ', data_temp[8].shape) #  (lead_time,)
    print('weight_map_thr  shape: ', data_temp[10].shape) # (points, forecast seq, 1)

    is_test_run = True
    is_test_plot = False

    if is_test_run:

        import random
        import torch

        manual_seed = 0
        random.seed(manual_seed)

        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        # the dataloader should accept (sampling points, seq length, feature)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=10,
                                                   shuffle=False,
                                                   pin_memory=False,
                                                   num_workers=1,
                                                   prefetch_factor=1)

        end = time.time()

        from tqdm import tqdm

        for i, x in tqdm(enumerate(train_loader), total=len(train_loader)):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))

            print(x[0].numpy().shape) # (batch*points, seq, features) 

            print(x[1].numpy().shape) # (batch*points, seq, features)

            print(x[2].numpy().shape) 

            print(x[3].numpy().shape) 

            print(x[4].numpy().shape) # (batch*points, seq, features)

            print(x[5].numpy().shape) # (batch*points, seq, features)

            print(x[6].numpy().shape) # (batch*points, seq, features)

            print(x[7].numpy().shape) # (batch*points, seq, features)

            print(x[8].numpy().shape) # (batch*points, seq, features)
            end = time.time()

    if is_test_plot: # visualize the target discharge

        matplotlib.use('Agg') # update the backend

        for i in range(len(dataset)): # i represents one day of the data
            i += 0
            (data_glofas, data_era5_land, data_static,
             data_thresholds, _,_) = dataset[int(i)] 

            print (data_era5_land.shape)
            print (data_thresholds.shape)

            x, y = np.meshgrid(np.arange(dataset.width), np.arange(dataset.height))
            x = x.flatten()[dataset.mask_valid == 1]
            y = y.flatten()[dataset.mask_valid == 1]
            x = x[dataset.random_indices]
            y = y[dataset.random_indices]

            background_image = dataset.mask_valid.copy().astype(np.float32)

            background_image[dataset.mask_valid == 0] = np.nan
            indices = np.arange(len(background_image))[dataset.mask_valid == 1]
            background_image[indices] = data_glofas[1, -1, :]
            background_image = background_image.reshape(dataset.height, dataset.width)

            import colormaps as cmaps

            plt.imshow(background_image, cmap=cmaps.turqw1_r)

            colors = ['bisque', 'crimson', 'goldenrod', 'red', 'cyan',
                      'peru', 'darkcyan', 'limegreen', 'navy', 'magenta']

            plt.axis('off')
            plt.savefig('/p/scratch/cjjsc39/zhang36/Neural_GloFAS/dataset/test_europe_%s.png' % i, dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()
