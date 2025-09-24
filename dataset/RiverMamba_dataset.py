# ------------------------------------------------------------------------------------------------------------
"""
Dataset class for RiverMamba

The dataset include: (a) Copernicus river discharge reanalysis data from the Global Flood Awareness System (GloFAS)
                      (b) GDRC observational river discharge data
                      (c) ECMWF HRES meteorological forecast
                      (d) CPC precipitation data
                      (e) ERA5-Land from ECMWF
                      (f) LISFLOOD / HydroRIVERS static data

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
import os
import json
from torch.utils.data import Dataset
import warnings
from datetime import datetime, timedelta
import time
import matplotlib
# matplotlib.use('TkAgg')

np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
import dask

dask.config.set(scheduler='synchronous')

# ------------------------------------------------------------------

class RiverMamba_Dataset(Dataset):
    """
        Dataset class for RiverMamba

        Attributes
        ----------
            root_glofas_reanalysis (str): directory to GloFAS reanalysis dataset
            root_era5_land_reanalysis (str): directory to ERA5-Land reanalysis dataset
            root_hres_forecast (str): directory to ECMWF-HRES forecast dataset
            root_static (str): directory to static dataset
            root_obs (str): directory to GRDC observational dataset
            root_cpc (str): directory to CPC precipitation dataset
            is_hres_forecast (bool, optional): option to use ECMWF-HRES as forecast, otherwise ERA5-Land will be used as forecast
            nan_fill (float): value to replace missing values
            delta_t (int): temporal resolution of the input data (hindcast)
            delta_t_f (int): temporal resolution of the forecast (lead time)
            is_shuffle (bool): option to shuffle the data. Defaults to False
            is_sample (bool): option to sample points. Defaults to True
            is_sample_aifas (bool): option to sample river diagnostic points from AIFAS. Defaults to True
            n_points (int): number of points in the dataset, if sample is True this is equivalent to the number of points to sample
            variables_glofas (list): list of GloFAS dynamic variable names
            variables_era5_land (list): list of ERA5-Land dynamic variable names
            variables_hres_forecast (list): list of ECMWF-HRES dynamic variable names
            variables_static (list): list of LISFLOOD/HydroRIVERS static variable names
            variables_cpc (list): list of CPC dynamic variable names
            variables_glofas_log1p (list): list of GloFAS dynamic variable names for the log1p transformation
            variables_era5_land_log1p (list): list of ERA5-Land dynamic variable names for the log1p transformation
            variables_hres_forecast_log1p (list): list of ECMWF-HRES dynamic variable names for the log1p transformation
            variables_static_log1p (list): list of LISFLOOD/HydroRIVERS static variable names for the log1p transformation
            variables_cpc_log1p (list): list of CPC dynamic variable names for the log1p transformation
            is_add_xyz (bool): option to add xyz coordinate to the static features
            curves (list): names of the curves to be used
            is_shuffle_curves (bool): option to shuffle the order of the curves
            is_norm (bool): option to normalize the data. Defaults to True
            years (list): list of years. Defaults to None
            lat_min (float): minimum latitude
            lat_max (float): maximum latitude
            lon_min (float): minimum longitude
            lon_max (float): maximum longitude
            is_obs (bool): option to use GRDC observational river discharge data as target
            alpha (float): alpha hyperparameter for the weights
            static_dataset (str): name of the static dataset either 'LISFLOOD' or 'HydroRIVERS'
            is_sample_curves (bool): option to sample points along the curve instead of random sampling
            is_val (bool): option to determine if the data is used for validation

        Methods
        -------

        __load_glofas_statistic()
            Private method to get the statistics of the GloFAS reanalysis dataset from the root directory
        __load_era5_land_statistic()
            Private method to get the statistics of the ERA5-Land dataset from the root directory
        __load_hres_statistic()
            Private method to get the statistics of the ECMWF-HRES dataset from the root directory
        __load_cpc_statistic()
            Private method to get the statistics of the cpc dataset from the root directory
        __load_static_statistic()
            Private method to get the statistics of the Static dataset from the root directory
        __get_var_n()
            Private method to get the number of variables
        __get_path()
            Private method to get the dataset files inside the root directory
        __load_weight_map()
            Private method to get the weight map from the root static directory
        __generate_leadtime_weight()
            Private method to generate the weights based on the lead time
        __load_flood_thresholds_obs()
            Private method to get the GRDC observational flood threshold maps (9 severity levels) from the root static directory.
            Flood thresholds are computed for selected return periods i.e., of 1.5, 2, 5, 10, 20, 50, 100, 200, and 500 years.
        __load_flood_thresholds()
            Private method to get the GloFAS reanalysis flood threshold maps (9 severity levels) from the root static directory.
            Flood thresholds are computed for selected return periods i.e., of 1.5, 2, 5, 10, 20, 50, 100, 200, and 500 years.
        __load_static_data()
            Private method to get the static data
        __load_mask_valid()
            Private method to get the mask for valid pixels from the root static directory
        __load_mask_obs()
            Private method to get the mask for the GRDC stations where the observations are available from the root static directory
        __load_curves()
            Private method to get the curves and the indices to sort the points
        min_max_scale()
            Helper method to normalize an array between new minimum and maximum values
        get_day_of_year()
            Helper method to get the day-of-the-year from the file name
        __load_dynamic_data()
            Private method to load NETCDF data from the files
        __load_hres_data()
            Private method to load ECMWF-HRES/ERA5-Land forecast data from the files
        log1p_transform()
            Helper method to transform an array by log1p transformation * sign(array)
        log1p_inv_transform()
            Helper method to inverse transform an array by expm1 transformation * sign(array)
        transform()
            Helper method to transform an array by mean and standard deviation
        inv_transform()
            Helper method to inversely transform an array by mean and standard deviation
        get_flood_thresholds()
            Helper method to get flood thresholds
        get_stations()
            method to get the mask of diagnostic points or stations
        generate_thr_weights()
            Helper method to generate weights based on the return periods and lead time of the reanalysis river discharge
        generate_thr_weights_obs()
            Helper method to generate weights based on the return periods and lead time of the observational river discharge
        __getitem__()
            Method to load datacube by the file index
        __len__()
            Method to get the number of samples in the dataset
    """

    def __init__(self,
                 root_glofas_reanalysis: str,
                 root_era5_land_reanalysis: str,
                 root_hres_forecast: str,
                 root_static: str,
                 root_obs: str,
                 root_cpc: str,
                 is_hres_forecast: bool = True,
                 nan_fill: float = 0.,
                 delta_t: int = 4,
                 delta_t_f: int = 15,
                 is_shuffle: bool = False,
                 is_sample: bool = True,
                 is_sample_aifas: bool = True,
                 n_points: int = 100000,
                 variables_glofas: list = None,
                 variables_era5_land: list = None,
                 variables_hres_forecast: list = None,
                 variables_static: list = None,
                 variables_cpc: list = None,
                 variables_glofas_log1p: list = None,
                 variables_era5_land_log1p: list = None,
                 variables_hres_forecast_log1p: list = None,
                 variables_static_log1p: list = None,
                 variables_cpc_log1p: list = None,
                 is_add_xyz: bool = False,
                 curves: list = None,
                 is_shuffle_curves: bool = False,
                 is_norm: bool = True,
                 years: list = None,
                 lat_min: float = None,
                 lat_max: float = None,
                 lon_min: float = None,
                 lon_max: float = None,
                 is_obs: bool = False,
                 alpha: float = 0.25,
                 static_dataset: str = 'LISFLOOD',
                 is_sample_curves: bool = True,
                 is_val: bool = False,
                 ):

        """
            Args:
            ----------
            root_glofas_reanalysis (str): directory to GloFAS reanalysis dataset
            root_era5_land_reanalysis (str): directory to ERA5-Land reanalysis dataset
            root_hres_forecast (str): directory to ECMWF-HRES forecast dataset
            root_static (str): directory to static dataset
            root_obs (str): directory to GRDC observational dataset
            root_cpc (str): directory to CPC precipitation dataset
            is_hres_forecast (bool, optional): option to use ECMWF-HRES as forecast, otherwise ERA5-Land will be used as forecast. Defaults to True
            nan_fill (float): value to replace missing values. Default to 0.
            delta_t (int, optional): temporal resolution of the input data (hindcast). Default to 4
            delta_t_f (int, optional): temporal resolution of the forecast (lead time). Default to 15
            is_shuffle (bool, optional): option to shuffle the data. Defaults to False
            is_sample (bool, optional): option to sample points. Defaults to True
            is_sample_aifas (bool, optional): option to sample river diagnostic points from AIFAS. Defaults to True
            n_points (int, optional): number of points in the dataset, if sample is True this is equivalent to the number of points to sample. Defaults to 100000
            variables_glofas (list, optional): list of GloFAS dynamic variable names. Defaults to None
            variables_era5_land (list, optional): list of ERA5-Land dynamic variable names. Defaults to None
            variables_hres_forecast (list, optional): list of ECMWF-HRES dynamic variable names. Defaults to None
            variables_static (list, optional): list of LISFLOOD/HydroRIVERS static variable names. Defaults to None
            variables_cpc (list, optional): list of CPC dynamic variable names. Defaults to None
            variables_glofas_log1p (list, optional): list of GloFAS dynamic variable names for the log1p transformation. Defaults to None
            variables_era5_land_log1p (list, optional): list of ERA5-Land dynamic variable names for the log1p transformation. Defaults to None
            variables_hres_forecast_log1p (list, optional): list of ECMWF-HRES dynamic variable names for the log1p transformation. Defaults to None
            variables_static_log1p (list, optional): list of LISFLOOD/HydroRIVERS static variable names for the log1p transformation. Defaults to None
            variables_cpc_log1p (list, optional): list of CPC dynamic variable names for the log1p transformation. Defaults to None
            is_add_xyz (bool, optional): option to add xyz coordinate to the static features. Defaults to False
            curves (list, optional): names of the curves to be used. Defaults to 'gilbert'
            is_shuffle_curves (bool, optional): option to shuffle the order of the curves. Defaults to False
            is_norm (bool, optional): option to normalize the data. Defaults to True
            years (list, optional): list of years. Defaults to None
            lat_min (float, optional): minimum latitude. Defaults to None
            lat_max (float, optional): maximum latitude. Defaults to None
            lon_min (float, optional): minimum longitude. Defaults to None
            lon_max (float, optional): maximum longitude. Defaults to None
            is_obs (bool, optional): option to use GRDC observational river discharge data as target. Defaults to False
            alpha (float, optional): alpha hyperparameter for the weights. Defaults to 0.25
            static_dataset (str, optional): name of the static dataset either 'LISFLOOD' or 'HydroRIVERS'. Defaults to LISFLOOD
            is_sample_curves (bool, optional): option to sample points along the curve instead of random sampling. Defaults to True
            is_val (bool, optional): option to determine if the data is used for validation. Defaults to False
        """

        super().__init__()

        self.root_glofas_reanalysis = root_glofas_reanalysis
        self.root_era5_land_reanalysis = root_era5_land_reanalysis
        self.root_hres_forecast = root_hres_forecast
        self.root_static = root_static
        self.root_obs = root_obs
        self.root_cpc = root_cpc

        self.nan_fill = nan_fill
        self.delta_t = delta_t
        self.delta_t_f = delta_t_f

        self.is_hres_forecast = is_hres_forecast
        self.is_shuffle = is_shuffle
        self.is_sample_aifas = is_sample_aifas
        self.is_sample = is_sample
        self.is_norm = is_norm
        self.years = years

        self.is_obs = is_obs
        self.alpha = alpha

        self.static_dataset = static_dataset
        self.is_sample_curves = is_sample_curves

        # variables are not sorted
        self.variables_glofas = variables_glofas
        self.variables_era5_land = variables_era5_land
        self.variables_hres_forecast = variables_hres_forecast
        self.variables_static = variables_static
        self.variables_glofas_log1p = variables_glofas_log1p
        self.variables_era5_land_log1p = variables_era5_land_log1p
        self.variables_hres_forecast_log1p = variables_hres_forecast_log1p
        self.variables_static_log1p = variables_static_log1p
        self.variables_cpc = variables_cpc
        self.variables_cpc_log1p = variables_cpc_log1p

        self.is_add_xyz = is_add_xyz

        if self.variables_glofas_log1p:
            self._variables_glofas_log1p_indices = [self.variables_glofas.index(v) for v in self.variables_glofas_log1p]
            self._variables_glofas_norm_indices = [x for x in range(len(self.variables_glofas)) if
                                                   x not in self._variables_glofas_log1p_indices]
        else:
            self._variables_glofas_norm_indices = [x for x in range(len(self.variables_glofas))]

        if self.variables_era5_land_log1p:
            self._variables_era5_land_log1p_indices = [self.variables_era5_land.index(v) for v in
                                                       self.variables_era5_land_log1p]
            self._variables_era5_land_norm_indices = [x for x in range(len(self.variables_era5_land)) if
                                                      x not in self._variables_era5_land_log1p_indices]
        else:
            self._variables_era5_land_norm_indices = [x for x in range(len(self.variables_era5_land))]

        if self.variables_hres_forecast_log1p:
            self._variables_hres_forecast_log1p_indices = [self.variables_hres_forecast.index(v) for v in
                                                           self.variables_hres_forecast_log1p]
            self._variables_hres_norm_indices = [x for x in range(len(self.variables_hres_forecast)) if
                                                 x not in self._variables_hres_forecast_log1p_indices]
        else:
            self._variables_hres_norm_indices = [x for x in range(len(self.variables_hres_forecast))]

        if self.variables_cpc_log1p:
            self._variables_cpc_log1p_indices = [self.variables_cpc.index(v) for v in self.variables_cpc_log1p]
            self._variables_cpc_norm_indices = [x for x in range(len(self.variables_cpc)) if
                                                x not in self.variables_cpc_log1p]
        else:
            self._variables_cpc_norm_indices = [x for x in range(len(self.variables_cpc))]

        if self.variables_static_log1p:
            self._variables_static_log1p_indices = [self.variables_static.index(v) for v in self.variables_static_log1p]

        # get river discharge index
        self.dis24_index = self.variables_glofas.index('dis24')

        self.curves = curves if curves else ['gilbert']
        self.is_shuffle_curves = is_shuffle_curves

        # GloFAS domain is until -60 S
        # only slice the images, point cloud data will not be indexed
        if lat_min is None:
            self.lat_min = -60
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
        self.__load_hres_statistic()
        self.__load_static_statistic()
        self.__load_cpc_statistic()
        self.__load_mask_valid()
        self.__load_mask_obs()

        if is_sample_aifas:
            indices_aifas = self.get_stations("AIFAS")
            self.indices_aifas = np.where(indices_aifas == 1)[0].tolist()

        self.n_points = n_points

        self.__load_weight_map()
        self.__generate_leadtime_weight()

        if self.is_obs:
            self.__load_flood_thresholds_obs()
        else:
            self.__load_flood_thresholds()

        self.__load_static_data()
        self.__load_curves()

        self.is_val = is_val

        if is_shuffle:
            np.random.shuffle(self.files)

    # TODO combine all statistics functions in one generic function
    def __load_glofas_statistic(self):
        """ Private method to get the statistics of the GloFAS reanalysis dataset from the root directory """

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

    def __load_era5_land_statistic(self):
        """ Private method to get the statistics of the ERA5-Land dataset from the root directory """

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
        """ Private method to get the statistics of the ECMWF-HRES dataset from the root directory """

        # statistics will be taken from ERA5-Land
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
        """ Private method to get the statistics of the cpc dataset from the root directory """

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
        """ Private method to get the statistics of the Static dataset from the root directory """

        if self.static_dataset == 'LISFLOOD':
            file_dataset = os.path.join(self.root_static, 'Static_LISFLOOD_statistics.json')
        elif self.static_dataset == 'HydroRIVERS':
            file_dataset = os.path.join(self.root_static, 'Static_HydroRIVERS_statistics.json')
        else:
            raise ValueError('Static dataset {} not supported'.format(self.static_dataset))

        with open(file_dataset, 'r') as file:
            dict_tmp = json.load(file)

            self.static_min, self.static_max, self.static_mean, self.static_std = [], [], [], []

            for v in self.variables_static:
                # if v in ['latitude', 'longitude', 'x', 'y', 'z']:
                #    continue

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
        """ Private method to get the number of variables """

        self.var_n_glofas = len(self.variables_glofas)
        self.var_n_era5_land = len(self.variables_era5_land)
        self.var_n_static = len(self.variables_static)
        self.var_n_hres = len(self.variables_hres_forecast)
        self.var_n_cpc = len(self.variables_cpc)

        if self.is_add_xyz:
            self.var_n_static += 3

        self.var_n = self.var_n_glofas + self.var_n_era5_land + self.var_n_static + self.var_n_hres + self.var_n_cpc

    def __get_path(self):
        """ Private method to get the dataset files inside the root directory """

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
                    # raise ValueError('year {} does not exist in the HRES data'.format(year))
                    warnings.warn(f'Year {year} does not exist in the HRES data. Using ERA5 forecast instead.')

            files = os.listdir(year_dir_glofas)
            files = [file for file in files if file.endswith('.nc')]
            files.sort()

            for file in files:
                day, month = file[6:8], file[4:6]
                files_glofas, files_era5_land, files_time, files_obs, files_cpc = [], [], [], [], []
                file_hres = None
                file_name = year + month + day

                for delta in reversed(range(self.delta_t)):

                    # shift glofas 1 day in the past
                    file_delta = (datetime(int(year), int(month), int(day)) - timedelta(days=delta + 1)).strftime(
                        '%Y%m%d')

                    file_glofas_delta = os.path.join(self.root_glofas_reanalysis, file_delta[:4], file_delta + '.nc')
                    if os.path.isfile(file_glofas_delta):
                        files_glofas.append(file_glofas_delta)
                    else:
                        warnings.warn('file {} does not exist in the GloFAS data'.format(file_glofas_delta))

                    files_time.append(self.get_day_of_year(file_delta))

                    # shift era5-land 1 day in the past
                    file_delta_era5 = (datetime(int(year), int(month), int(day)) - timedelta(days=delta + 1)).strftime(
                        '%Y%m%d')

                    file_era5_land_delta = os.path.join(self.root_era5_land_reanalysis, file_delta_era5[:4],
                                                        file_delta_era5 + '.nc')

                    if os.path.isfile(file_era5_land_delta):
                        files_era5_land.append(file_era5_land_delta)
                    else:
                        warnings.warn('file {} does not exist in the ERA5-Land data'.format(file_era5_land_delta))

                    # shift cpc 2 days in the past
                    file_delta_cpc = (datetime(int(year), int(month), int(day)) - timedelta(days=delta + 2)).strftime(
                        '%Y%m%d')

                    file_cpc_delta = os.path.join(self.root_cpc, file_delta_cpc[:4], file_delta_cpc + '.nc')

                    if os.path.isfile(file_cpc_delta):
                        files_cpc.append(file_cpc_delta)
                    else:
                        warnings.warn('file {} does not exist in the CPC data'.format(file_cpc_delta))

                for delta in range(self.delta_t_f):

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

                    # TODO add obs as an option
                    file_obs_delta = os.path.join(self.root_obs, file_delta[:4], file_delta + '.nc')

                    if os.path.isfile(file_obs_delta):
                        files_obs.append(file_obs_delta)
                    else:
                        warnings.warn('file {} does not exist in the observation data'.format(file_obs_delta))

                    if self.is_hres_forecast and delta == 1:  # because hres has all time steps in the last day
                        file_hres = os.path.join(self.root_hres_forecast, file[:4], file)

                        if not os.path.isfile(file_hres):
                            file_hres = None

                    files_time.append(self.get_day_of_year(file_delta))

                if file_hres is None:
                    forecast_type = 'era5'
                else:
                    forecast_type = 'hres'

                if len(files_glofas) == len(files_era5_land) == (self.delta_t + self.delta_t_f) and len(
                        files_obs) == self.delta_t_f and len(files_cpc) == self.delta_t:

                    if self.is_hres_forecast:
                        self.files.append({'glofas': files_glofas[:self.delta_t],
                                           'era5': files_era5_land[:self.delta_t],
                                           'glofas_target': files_glofas[self.delta_t:],
                                           'era5_forecast': files_era5_land[self.delta_t:],
                                           'hres_forecast': [file_hres] if file_hres else [],
                                           'cpc': files_cpc,
                                           'obs_target': files_obs,
                                           'file_name': file_name,
                                           'forecast_type': forecast_type
                                           })

                    else:
                        self.files.append({'glofas': files_glofas[:self.delta_t],
                                           'era5': files_era5_land[:self.delta_t],
                                           'glofas_target': files_glofas[self.delta_t:],
                                           'era5_forecast': files_era5_land[self.delta_t:],
                                           'cpc': files_cpc,
                                           'obs_target': files_obs,
                                           'file_name': file_name,
                                           'forecast_type': forecast_type
                                           })

                    self.data_time.append(files_time[:self.delta_t])

        # if needed
        self.data_time = np.array(self.data_time).astype(np.int16)

        if len(self.files) == 0:
            raise ValueError('No files were found in the root directories')

    def __load_weight_map(self):
        """ Private method to get the weight map from the root static directory """

        # load predefined weights
        self.weight_map = xr.open_dataset(os.path.join(self.root_static, "weight_maps/weight_map.nc"))
        self.weight_map = self.weight_map.sel(latitude=slice(self.lat_max, self.lat_min),
                                              longitude=slice(self.lon_min, self.lon_max))
        self.weight_map = self.weight_map['weight_map'].values
        #self.ww = self.weight_map.copy()  # just for visualization

        self.weight_map = self.weight_map.flatten()[self.mask_valid == 1]

        #self.random_indices = np.arange(len(self.weight_map))

        if self.is_sample_aifas:
            self.weight_map = self.weight_map[self.indices_aifas]

    def __generate_leadtime_weight(self):
        """ Private method to generate the weights based on the lead time """
        weights_t = np.exp(np.abs(np.arange(1, self.delta_t_f + 1) - self.delta_t_f - 1) * self.alpha)
        self.weights_t = weights_t[None, :, None]

    def __load_flood_thresholds_obs(self):
        """
        Private method to get the GRDC observational flood threshold maps (9 severity levels) from the root static directory.
        Flood thresholds are computed for selected return periods i.e., of 1.5, 2, 5, 10, 20, 50, 100, 200, and 500 years.
        """

        files_thr = ['flood_threshold_obs_rl_1.5', 'flood_threshold_obs_rl_2.0',
                     'flood_threshold_obs_rl_5.0', 'flood_threshold_obs_rl_10.0',
                     'flood_threshold_obs_rl_20.0', 'flood_threshold_obs_rl_50.0',
                     'flood_threshold_obs_rl_100.0', 'flood_threshold_obs_rl_200.0',
                     'flood_threshold_obs_rl_500.0', ]

        # 'flood_threshold_glofas_v4_mu', 'flood_threshold_glofas_v4_sigma']

        files_thr = [xr.open_dataset(os.path.join(self.root_static, "threshold_obs/grdc/" + file + ".nc")) for file in
                     files_thr]

        self._flood_thresholds_obs = xr.merge(files_thr)

        self._flood_thresholds_obs = self._flood_thresholds_obs.sel(latitude=slice(self.lat_max, self.lat_min),
                                                                    longitude=slice(self.lon_min, self.lon_max))
        self._flood_thresholds_obs = self._flood_thresholds_obs.to_array().values

        self._thr_obs = [1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

        N, H, W = self._flood_thresholds_obs.shape

        self._flood_thresholds_obs = self._flood_thresholds_obs.reshape(N, H * W)[:, self.mask_valid == 1]

        if self.is_sample_aifas:
            self.thresholds_obs = np.moveaxis(self._flood_thresholds_obs[:, self.indices_aifas], 0, -1)
        else:
            self.thresholds_obs = np.moveaxis(self._flood_thresholds_obs, 0, -1)

    def __load_flood_thresholds(self):
        """
        Private method to get the GloFAS reanalysis flood threshold maps (9 severity levels) from the root static directory.
        Flood thresholds are computed for selected return periods i.e., of 1.5, 2, 5, 10, 20, 50, 100, 200, and 500 years.
        """

        files_thr = ['flood_threshold_glofas_v4_rl_1.5', 'flood_threshold_glofas_v4_rl_2.0',
                     'flood_threshold_glofas_v4_rl_5.0', 'flood_threshold_glofas_v4_rl_10.0',
                     'flood_threshold_glofas_v4_rl_20.0', 'flood_threshold_glofas_v4_rl_50.0',
                     'flood_threshold_glofas_v4_rl_100.0', 'flood_threshold_glofas_v4_rl_200.0',
                     'flood_threshold_glofas_v4_rl_500.0', ]
        # 'flood_threshold_glofas_v4_mu', 'flood_threshold_glofas_v4_sigma']

        files_thr = [xr.open_dataset(os.path.join(self.root_static, "threshold/" + file + ".nc")) for file in files_thr]

        self._flood_thresholds = xr.merge(files_thr)
        self._flood_thresholds = self._flood_thresholds.sel(lat=slice(self.lat_max, self.lat_min),
                                                            lon=slice(self.lon_min, self.lon_max))
        self._flood_thresholds = self._flood_thresholds.to_array().values

        self._thr = [1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

        N, H, W = self._flood_thresholds.shape

        self._flood_thresholds = self._flood_thresholds.reshape(N, H * W)[:, self.mask_valid == 1]

        if self.is_sample_aifas:
            self.thresholds = np.moveaxis(self._flood_thresholds[:, self.indices_aifas], 0, -1)
        else:
            self.thresholds = np.moveaxis(self._flood_thresholds, 0, -1)

    def __load_static_data(self):
        """ Private method to get the static data """

        # these predefined variables are just for normalization
        variables_fraction = ['crp_pc_cse', 'crp_pc_use', 'for_pc_cse', 'for_pc_use', 'gla_pc_cse', 'gla_pc_use',
                              'cly_pc_cav', 'cly_pc_uav', 'glc_pc_c01', 'glc_pc_c02', 'glc_pc_c03', 'glc_pc_c04',
                              'glc_pc_c05', 'glc_pc_c06', 'glc_pc_c07', 'glc_pc_c08', 'glc_pc_c09', 'glc_pc_c10',
                              'glc_pc_c11', 'glc_pc_c12', 'glc_pc_c13', 'glc_pc_c14', 'glc_pc_c15', 'glc_pc_c16',
                              'glc_pc_c17', 'glc_pc_c18', 'glc_pc_c19', 'glc_pc_c20', 'glc_pc_c21', 'glc_pc_c22',
                              'glc_pc_u01', 'glc_pc_u02', 'glc_pc_u03', 'glc_pc_u04', 'glc_pc_u05', 'glc_pc_u06',
                              'glc_pc_u07', 'glc_pc_u08', 'glc_pc_u09', 'glc_pc_u10', 'glc_pc_u11', 'glc_pc_u12',
                              'glc_pc_u13', 'glc_pc_u14', 'glc_pc_u15', 'glc_pc_u16', 'glc_pc_u17', 'glc_pc_u18',
                              'glc_pc_u19', 'glc_pc_u20', 'glc_pc_u21', 'glc_pc_u22', 'glwd_v2_delta_area_pct',
                              'inu_pc_clt', 'inu_pc_cmn', 'inu_pc_cmx', 'inu_pc_ult', 'inu_pc_umn', 'inu_pc_umx',
                              'ire_pc_cse', 'ire_pc_use', 'kar_pc_cse', 'kar_pc_use', 'pac_pc_cse', 'pac_pc_use',
                              'pnv_pc_c01', 'pnv_pc_c02', 'pnv_pc_c03', 'pnv_pc_c04', 'pnv_pc_c05', 'pnv_pc_c06',
                              'pnv_pc_c07', 'pnv_pc_c08', 'pnv_pc_c09', 'pnv_pc_c10', 'pnv_pc_c11', 'pnv_pc_c12',
                              'pnv_pc_c13', 'pnv_pc_c14', 'pnv_pc_c15', 'pnv_pc_u01', 'pnv_pc_u02', 'pnv_pc_u03',
                              'pnv_pc_u04', 'pnv_pc_u05', 'pnv_pc_u06', 'pnv_pc_u07', 'pnv_pc_u08', 'pnv_pc_u09',
                              'pnv_pc_u10', 'pnv_pc_u11', 'pnv_pc_u12', 'pnv_pc_u13', 'pnv_pc_u14', 'pnv_pc_u15',
                              'prm_pc_cse', 'prm_pc_use', 'pst_pc_cse', 'pst_pc_use', 'slt_pc_cav', 'snd_pc_cav',
                              'snd_pc_uav', 'snw_pc_c01', 'snw_pc_c02', 'snw_pc_c03', 'snw_pc_c04', 'snw_pc_c05',
                              'snw_pc_c06', 'snw_pc_c07', 'snw_pc_c08', 'snw_pc_c09', 'snw_pc_c10', 'snw_pc_c11',
                              'snw_pc_c12', 'snw_pc_cmx', 'snw_pc_cyr', 'snw_pc_uyr', 'swc_pc_c01', 'swc_pc_c02',
                              'swc_pc_c03', 'swc_pc_c04', 'swc_pc_c05', 'swc_pc_c06', 'swc_pc_c07', 'swc_pc_c08',
                              'swc_pc_c09', 'swc_pc_c10', 'swc_pc_c11', 'swc_pc_c12', 'swc_pc_cyr', 'swc_pc_uyr',
                              'urb_pc_cse', 'urb_pc_use', 'wet_pc_c01', 'wet_pc_c02', 'wet_pc_c03', 'wet_pc_c04',
                              'wet_pc_c05', 'wet_pc_c06', 'wet_pc_c07', 'wet_pc_c08', 'wet_pc_c09', 'wet_pc_cg1',
                              'wet_pc_cg2', 'wet_pc_u01', 'wet_pc_u02', 'wet_pc_u03', 'wet_pc_u04', 'wet_pc_u05',
                              'wet_pc_u06', 'wet_pc_u07', 'wet_pc_u08', 'wet_pc_u09', 'wet_pc_ug1', 'wet_pc_ug2',
                              'dor_pc_pva', 'lka_pc_cse', 'lka_pc_use', 'annualSnowFraction_fs']

        if self.static_dataset == 'LISFLOOD':
            file_dataset = os.path.join(self.root_static, 'NeuralFAS_LISFLOOD_static.nc')
        elif self.static_dataset == 'HydroRIVERS':
            file_dataset = os.path.join(self.root_static, 'NeuralFAS_HydroRIVERS_static.nc')
        else:
            raise ValueError('Static dataset {} not supported'.format(self.static_dataset))

        with xr.open_dataset(file_dataset) as dataset:

            dataset = dataset.sel(latitude=slice(self.lat_max, self.lat_min),
                                  longitude=slice(self.lon_min, self.lon_max))

            self.data_static = dataset[self.variables_static]

            self.data_static = self.data_static.to_array().values

            _, self.height, self.width = self.data_static.shape

            if self.variables_static_log1p:
                self.data_static[self._variables_static_log1p_indices] = self.log1p_transform(
                    self.data_static[self._variables_static_log1p_indices])

            if self.is_norm:
                for v in range(len(self.static_mean)):

                    if self.variables_static[v] in variables_fraction:
                        self.data_static[v][np.isnan(self.data_static[v])] = 0
                    else:
                        self.data_static[v][np.isnan(self.data_static[v])] = self.static_min[v]

                    # hard codded normalization for the static features between -1 and +1
                    # self.data_static[v] = (self.data_static[v] - self.static_mean[v]) / (self.static_std[v] + 1e-6)
                    self.data_static[v] = self.min_max_scale(self.data_static[v],
                                                             self.static_min[v],
                                                             self.static_max[v],
                                                             -1,
                                                             1
                                                             )

        if self.is_add_xyz:
            # add xyz based on the WGS ellipsoid
            latitude_2d, longitude_2d = xr.broadcast(dataset.latitude, dataset.longitude)
            latitude_2d, longitude_2d = latitude_2d.values[None, :, :], longitude_2d.values[None, :, :]

            a = 6378137 / 1000
            b = 6356752.314245 / 1000
            e_2 = (a ** 2 - b ** 2) / a ** 2
            N = a / np.sqrt(1 - e_2 * np.sin(latitude_2d * np.pi / 180) ** 2)
            if self.static_dataset == 'LISFLOOD':
                h = dataset['elv'].values / 1000
            elif self.static_dataset == 'HydroRIVERS':
                h = dataset['elevation'].values / 1000

            x = (N + h) * np.cos(latitude_2d * np.pi / 180) * np.cos(longitude_2d * np.pi / 180)
            y = (N + h) * np.cos(latitude_2d * np.pi / 180) * np.sin(longitude_2d * np.pi / 180)
            z = ((1 - e_2) * N + h) * np.sin(latitude_2d * np.pi / 180)

            x = self.min_max_scale(x, np.nanmin(x), np.nanmax(x), -1, 1)
            y = self.min_max_scale(y, np.nanmin(y), np.nanmax(y), -1, 1)
            z = self.min_max_scale(z, np.nanmin(z), np.nanmax(z), -1, 1)

            self.data_static = np.concatenate((self.data_static, x, y, z), axis=0)

            self.variables_static = self.variables_static + ['x', 'y', 'z']

        self.data_static[np.isnan(self.data_static)] = self.nan_fill

        # import matplotlib
        # matplotlib.use('TkAgg')
        # for v in range(len(self.variables_static)):
        #    print(self.variables_static[v])
        #    print(np.min(self.data_static[v]), np.max(self.data_static[v]))
        #    plt.imshow(self.data_static[v])
        #    plt.show()

        self.data_static = self.data_static.reshape(self.var_n_static,
                                                    self.height * self.width)[:, self.mask_valid == 1].astype(
            np.float32)

        if self.is_sample_aifas:
            self.data_static = self.data_static[:, self.indices_aifas]

        self.data_static = np.moveaxis(self.data_static, 0, -1)

    def __load_mask_valid(self):
        """ Private method to get the mask for valid pixels from the root static directory """

        self.mask_valid = xr.open_dataset(os.path.join(self.root_static, "masks/mask_valid.nc"))
        #self.mask_valid = self.mask_valid.sel(latitude=slice(self.lat_max, self.lat_min),
        #                                      longitude=slice(self.lon_min, self.lon_max))
        self.mask_valid = self.mask_valid['mask_valid'].values.flatten()

        self.random_indices = np.arange(len(self.mask_valid))

    def __load_mask_obs(self):
        """ Private method to get the mask for the GRDC stations where the observations are available from the root static directory """

        indices_obs = self.get_stations("GRDC_obs")
        self.indices_obs = np.where(indices_obs == 1)[0].tolist()

    def __load_curves(self):
        """ Private method to get the curves and the indices to sort the points """

        # TODO optimize and remove unnecessary steps

        self.full_series = np.arange(self.n_points).astype(np.int64).tolist()

        self._n_valid_points = int(np.nansum(self.mask_valid))
        self._full_series = np.arange(self._n_valid_points)
        self.n_curves = len(self.curves)

        # self.curves_indices: [Number of curves, index and inverse index, Number of Points]
        self.curves_indices = np.zeros((self.n_curves, 2, self._n_valid_points), dtype=np.int32)

        for c, curve in enumerate(self.curves):
            # read the curve
            curve_idx = np.load(os.path.join(self.root_static, 'curves/' + curve + '.npy'))
            # curve_idx = np.load(os.path.join(self.root_static, 'curves_europe/' + curve + '.npy'))

            curve_idx = curve_idx.flatten()
            # remove invalid pixels
            curve_idx = curve_idx[self.mask_valid == 1]
            # generate regular series
            # curve_idx_data = np.arange(len(curve_idx))
            curve_idx = curve_idx.argsort()

            self.curves_indices[c, 0, :] = curve_idx

            # compute inverse indexing
            dictionary = dict(zip(curve_idx, self._full_series))
            self.curves_indices[c, 1, :] = list(map(dictionary.get, self._full_series))

        # self.sampled_series = np.arange(self.n_points)

        # """
        # always sample stations when random sampling is True
        if self.is_sample or self.is_sample_aifas:
            self.sampled_series = np.arange(self.n_points)
            # self.sampled_series = np.arange(self.n_points - len(self.indices_obs))
            self.n_points_wo_obs = self.n_points - len(self.indices_obs)

            if self.is_sample_aifas:
                # self.indices_aifas_wo_obs = np.delete(self.indices_aifas, self.indices_obs)
                # self.indices_aifas_wo_obs =[i for i in self.indices_aifas if i not in self.indices_obs]
                # m = ~np.isin(self.indices_aifas, self.indices_obs, invert=False, assume_unique=True)
                # self.indices_aifas_wo_obs = self.indices_aifas[m]
                self.indices_aifas_wo_obs = np.setdiff1d(self.indices_aifas, self.indices_obs)
            else:
                self.indices_all_wo_obs = np.setdiff1d(self._full_series, self.indices_obs)
        # """

        # next are to sample points along the curves
        if self.is_sample_aifas:

            sampled_series = np.arange(len(self.indices_aifas))
            self._n_valid_points_aifas = len(self.indices_aifas)

            random_indices = self.indices_aifas

            dictionary = dict(zip(random_indices, sampled_series))
            curves = []
            for curve, _ in self.curves_indices:
                curve_i = list(map(dictionary.get, curve[np.isin(curve, random_indices, assume_unique=True)]))
                dictionary_i = dict(zip(curve_i, sampled_series))
                curves.append([curve_i, list(map(dictionary_i.get, sampled_series))])

            curves_series = np.array(curves).astype(np.int32)[:, 0, :]  # number of curves, P

            self.curves_series = curves_series.copy()

            for c in range(self.n_curves):
                self.curves_series[c] = np.array(self.indices_aifas)[curves_series[c].tolist()]

            # TODO optimize this function
            self.sample_division = int(np.round(self._n_valid_points_aifas / self.n_points))
            self.samples_along_curve = []
            self.samples_along_curve_static = []
            for r in range(self.sample_division):
                for c in range(self.n_curves):
                    ll = self.curves_series[c][r::self.sample_division]
                    ll = np.append(self.indices_obs, np.setdiff1d(ll, self.indices_obs))[:self.n_points]

                    if len(ll) < self.n_points:
                        l_add = np.random.choice(np.setdiff1d(self.indices_aifas_wo_obs, ll),
                                                 self.n_points - len(ll),
                                                 replace=False)
                        ll = np.append(ll, l_add)

                    ll.sort()

                    self.samples_along_curve.append(ll)

                    st = set(ll)
                    static_indices = [i for i, p in enumerate(self.indices_aifas) if p in st]
                    self.samples_along_curve_static.append(static_indices)

            self.n_samples_along_curve = len(self.samples_along_curve)
        else:

            curves_series = self.curves_indices.astype(np.int32)[:, 0, :]  # number of curves, P
            self.curves_series = curves_series.copy()

            # for c in range(self.n_curves):
            # self.curves_series[c] = np.array(self.indices_aifas)[curves_series[c].tolist()]

            # TODO optimize this function
            self.sample_division = int(np.round(self._n_valid_points / self.n_points))
            self.samples_along_curve = []
            self.samples_along_curve_static = []
            for r in range(self.sample_division):
                for c in range(self.n_curves):
                    ll = self.curves_series[c][r::self.sample_division]
                    ll = np.append(self.indices_obs, np.setdiff1d(ll, self.indices_obs))[:self.n_points]

                    if len(ll) < self.n_points:
                        l_add = np.random.choice(np.setdiff1d(self.indices_aifas_wo_obs, ll),
                                                 self.n_points - len(ll),
                                                 replace=False)
                        ll = np.append(ll, l_add)

                    ll.sort()

                    self.samples_along_curve.append(ll)

                    st = set(ll)
                    static_indices = [i for i, p in enumerate(self._full_series) if p in st]
                    self.samples_along_curve_static.append(static_indices)

            self.n_samples_along_curve = len(self.samples_along_curve)

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
        Helper method to get the day-of-the-year from the file name

        Parameters
        ----------
        file : str
          name of the file in the dataset

        Returns
        ----------
        day-of-the-year : int
            corresponding day number of the file
        """

        """
        https://stackoverflow.com/questions/620305/convert-year-month-day-to-day-of-year-in-python 
        given year, month, day return day of year Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 
        """

        file_name = os.path.splitext(os.path.basename(os.path.normpath(file)))[0]

        # year = int(file_name[:4])
        month = int(file_name[4:6])
        day = int(file_name[6:])

        # return datetime(year, month, day).timetuple().tm_yday
        return int((275 * month) / 9.0) - 1 * int((month + 9) / 12.0) + day - 30

    def __load_dynamic_data(self, NetCDF_files, dynamic_variables):
        """ Private method to load NETCDF data from the files """

        datacube = xr.open_mfdataset(NetCDF_files,
                                     combine='nested',
                                     concat_dim='None',
                                     # preprocess=lambda x: x[dynamic_variables],
                                     parallel=False,
                                     engine='netcdf4',
                                     )[dynamic_variables].to_array().values

        return datacube

    def __load_hres_data(self, NetCDF_files, dynamic_variables, lead_time, forecast_type='hres'):
        """ Private method to load ECMWF-HRES/ERA5-Land forecast data from the files """

        if forecast_type == 'hres':
            datacube = xr.open_dataset(NetCDF_files[0]).isel(step=lead_time)[dynamic_variables].to_array().values
        else:
            if isinstance(lead_time, int):
                NetCDF_files = NetCDF_files[lead_time]
                datacube = xr.open_dataset(NetCDF_files)[dynamic_variables].to_array().values
            else:
                datacube = xr.open_mfdataset(NetCDF_files,
                                             combine='nested',
                                             concat_dim='None',
                                             # preprocess=lambda x: x[dynamic_variables],
                                             parallel=False,
                                             engine='netcdf4',
                                             )[dynamic_variables].to_array().values
        return datacube

    def log1p_transform(self, x):
        """ Helper method to transform an array by log1p transformation * sign(array) """
        return np.sign(x) * np.log1p(np.abs(x))

    def log1p_inv_transform(self, x):
        """ Helper method to inverse transform an array by expm1 transformation * sign(array) """
        return np.sign(x) * np.expm1(np.abs(x))

    def transform(self, x: np.ndarray, mean, std) -> np.ndarray:
        """ Helper method to transform an array by mean and standard deviation """
        x_norm = (x - mean) / (std + 1e-6)
        return x_norm

    def inv_transform(self, x_norm: np.ndarray, mean, std) -> np.ndarray:
        """ Helper method to inversely transform an array by mean and standard deviation """
        x = (x_norm * (std + 1e-6)) + mean
        return x

    def get_flood_thresholds(self, thr: list = None):
        """ Helper method to get flood thresholds """

        if thr is None:
            thr = self._thr
        assert set(thr).issubset(self._thr), "possible thresholds are 1.5, 2, 5, 10, 20, 50, 100, 200, 500"
        return self._flood_thresholds[[self._thr.index(item) for item in thr]]

    def get_stations(self, dataset='AIFAS'):

        """ method to get the mask of diagnostic points or stations """
        assert dataset in ['AIFAS', 'GRDC_obs']

        mask_station = xr.open_dataset(os.path.join(self.root_static, "masks/mask_{}_points.nc".format(dataset)))
        #mask_station = mask_station.sel(latitude=slice(self.lat_max, self.lat_min),
        #                                longitude=slice(self.lon_min, self.lon_max))
        mask_station = mask_station['mask_points'].values.flatten()[self.mask_valid == 1]

        return mask_station

    def generate_thr_weights(self, target_glofas, thresholds):
        """ Helper method to generate weights based on the return periods and lead time of the reanalysis river discharge """

        # target_glofas  1, 7, P
        # threshold P, 9
        target_glofas = np.repeat(target_glofas[:, :, :, None], 9, axis=-1)

        weight_map_thr = np.max((target_glofas >= thresholds) * self._thr, axis=-1)
        weight_map_thr[weight_map_thr == 0.] = 1

        weight_map_thr = weight_map_thr * self.weights_t

        return np.moveaxis(weight_map_thr, 0, -1)

    def generate_thr_weights_obs(self, target_glofas, thresholds_obs):
        """ Helper method to generate weights based on the return periods and lead time of the observational river discharge """
        # target_glofas  1, 7, P
        # threshold P, 9
        target_glofas = np.repeat(target_glofas[:, :, :, None], 9, axis=-1)

        weight_map_thr = np.max((target_glofas >= thresholds_obs) * self._thr_obs, axis=-1)
        weight_map_thr[weight_map_thr == 0.] = 1

        weight_map_thr = weight_map_thr * self.weights_t

        return np.moveaxis(weight_map_thr, 0, -1)

    def __getitem__(self, index):
        """
        Method to load datacube by the file index

        Args:
        ----------
        index (int): the index of the file

        Returns
        ----------
        glofas (np.array): GloFAS reanalysis dynamic data [T, P, V]
        era5 (np.array): ERA5-Land reanalysis dynamic data [T, P, V]
        hres_forecast (np.array): ECMWF-HRES meteorological forecast [lead time, P, V]
        static (np.array): static data [P, V]
        cpc (np.array): cpc precipitation data [T, P, V]
        glofas_target (np.array): river discharge target data [lead time, P, 1]
        obs_target (np.array): river discharge target observational data [lead time, P, 1]
        weight (np.array): weights for each point based on the return periods and lead time [lead time, P, 1]
        curves (np.array): curves indices [number of curves, 2, P]
                           2nd dimension represents the direction of the mapping, first element is the order of the
                           points to form the curve,
                           the second element along the 2nd dimension is the mapping to inverse the curve
        file_name (str): name of the file
        """

        files_glofas, files_era5_land, files_glofas_t, files_cpc, files_obs_t = (self.files[index]['glofas'],
                                                                                 self.files[index]['era5'],
                                                                                 self.files[index]['glofas_target'],
                                                                                 self.files[index]['cpc'],
                                                                                 self.files[index]['obs_target']
                                                                                 )

        forecast_type = self.files[index]['forecast_type']  # Get forecast type
        files_hres = self.files[index]['hres_forecast'] if forecast_type == 'hres' \
            else self.files[index]['era5_forecast']

        # if sample diagnostic river points
        if self.is_sample_aifas:
            # if sample a random subset of the diagnostic river points
            if self.is_sample:
                # whether to sample along the curve or sample randomly sparse points
                if not self.is_sample_curves:
                    if np.random.choice(2):  # sample every n points along the curve
                        r = np.random.choice(self.n_samples_along_curve)
                        random_indices = self.samples_along_curve[r]
                        data_static = self.data_static[self.samples_along_curve_static[r], :]  # P, V

                        if self.is_obs:
                            thresholds_obs = self.thresholds_obs[self.samples_along_curve_static[r], :]  # P, V
                        else:
                            thresholds = self.thresholds[self.samples_along_curve_static[r], :]  # P, V

                        # data_weight = self.weight_map[self.samples_along_curve_static[r]]  # P
                    else:  # sample n points randomly
                        random_indices = np.append(np.random.choice(self.indices_aifas_wo_obs,
                                                                    self.n_points_wo_obs,
                                                                    replace=False),
                                                   self.indices_obs)
                        random_indices.sort()

                        st = set(random_indices)
                        static_indices = [i for i, p in enumerate(self.indices_aifas) if p in st]
                        data_static = self.data_static[static_indices, :]  # P, V
                        if self.is_obs:
                            thresholds_obs = self.thresholds_obs[static_indices, :]  # P, V
                        else:
                            thresholds = self.thresholds[static_indices, :]  # P, V
                        # data_weight = self.weight_map[static_indices]  # P, V
                else:   # sample n connected points along the curve
                    if self.is_val:  # if validation or test sample predefined segment along the curve
                        r = np.random.choice(self.sample_division)
                        random_indices = self.curves_series[2, :][self.n_points * r: self.n_points * (r + 1)]
                    else:  # if training sample a random segment along the curve
                        r = np.random.choice(self._n_valid_points_aifas - self.n_points + 1, 1)[0]
                        random_indices = self.curves_series[2, :][np.arange(self.n_points) + r]

                    random_indices.sort()

                    st = set(random_indices)
                    static_indices = [i for i, p in enumerate(self.indices_aifas) if p in st]
                    data_static = self.data_static[static_indices, :]  # P, V
                    if self.is_obs:
                        thresholds_obs = self.thresholds_obs[static_indices, :]  # P, V
                    else:
                        thresholds = self.thresholds[static_indices, :]  # P, V
                    # data_weight = self.weight_map[static_indices]  # P, V

            else:  # get all aifas diagnostic river points
                random_indices = self.indices_aifas.copy()
                data_static = self.data_static

                if self.is_obs:
                    thresholds_obs = self.thresholds_obs
                else:
                    thresholds = self.thresholds

                # data_weight = self.weight_map

            dictionary = dict(zip(random_indices, self.sampled_series))
            curves = []
            for curve, _ in self.curves_indices:
                curve_i = list(map(dictionary.get, curve[np.isin(curve, random_indices, assume_unique=True)]))
                dictionary_i = dict(zip(curve_i, self.sampled_series))
                curves.append([curve_i, list(map(dictionary_i.get, self.sampled_series))])

            curves = np.array(curves, dtype=np.int32)  # .astype(np.int32)  # number of curves, P

            data_glofas = self.__load_dynamic_data(files_glofas, self.variables_glofas)[:, :, random_indices]  # V, T, P
            data_era5_land = self.__load_dynamic_data(files_era5_land, self.variables_era5_land)[:, :,
                             random_indices]  # V, T, P
            data_cpc = self.__load_dynamic_data(files_cpc, self.variables_cpc)[:, :, random_indices]  # 1, Tf, P
            target_glofas = self.__load_dynamic_data(files_glofas_t, ['dis24'])[:, :, random_indices]  # 1, Tf, P
            target_obs = self.__load_dynamic_data(files_obs_t, ['dis24']).astype(
                np.float32)[:, :, random_indices]  # 1, Tf, P

            data_hres = self.__load_hres_data(files_hres,
                                              self.variables_hres_forecast,
                                              np.arange(0, self.delta_t_f), forecast_type)[:, :,
                        random_indices]  # 1, Tf, P

            if self.is_obs:
                weight_map_thr = self.generate_thr_weights_obs(target_obs, thresholds_obs)
            else:
                weight_map_thr = self.generate_thr_weights(target_glofas, thresholds)

        else:  # don't sample aifas diagnostic river points
            if self.is_sample:  # sample points from the full resolution
                # whether to sample along the curve or sample randomly sparse points
                if not self.is_sample_curves:
                    if np.random.choice(2):   # sample every n points along the curve
                        r = np.random.choice(self.n_samples_along_curve)
                        random_indices = self.samples_along_curve[r]
                        data_static = self.data_static[self.samples_along_curve_static[r], :]  # P, V

                        if self.is_obs:
                            thresholds_obs = self.thresholds_obs[self.samples_along_curve_static[r], :]  # P, V
                        else:
                            thresholds = self.thresholds[self.samples_along_curve_static[r], :]  # P, V

                        # data_weight = self.weight_map[self.samples_along_curve_static[r]]  # P
                    else:  # sample n points randomly
                        random_indices = np.append(np.random.choice(self.indices_all_wo_obs,
                                                                    self.n_points_wo_obs,
                                                                    replace=False),
                                                   self.indices_obs)
                        random_indices.sort()

                        st = set(random_indices)
                        static_indices = [i for i, p in enumerate(self._full_series) if p in st]
                        data_static = self.data_static[static_indices, :]  # P, V
                        if self.is_obs:
                            thresholds_obs = self.thresholds_obs[static_indices, :]  # P, V
                        else:
                            thresholds = self.thresholds[static_indices, :]  # P, V
                        # data_weight = self.weight_map[static_indices]  # P, V

                else:   # sample n connected points along the curve
                    if self.is_val:   # if validation or test sample predefined segment along the curve
                        r = np.random.choice(self.sample_division)
                        random_indices = self.curves_series[2, :][self.n_points * r: self.n_points * (r + 1)]
                    else:   # if training sample a random segment along the curve
                        r = np.random.choice(self._n_valid_points - self.n_points + 1, 1)[0]
                        random_indices = self.curves_series[2, :][np.arange(self.n_points) + r]

                    random_indices.sort()

                    st = set(random_indices)
                    static_indices = [i for i, p in enumerate(self._full_series) if p in st]
                    data_static = self.data_static[static_indices, :]  # P, V
                    if self.is_obs:
                        thresholds_obs = self.thresholds_obs[static_indices, :]  # P, V
                    else:
                        thresholds = self.thresholds[static_indices, :]  # P, V
                    # data_weight = self.weight_map[static_indices]  # P, V

                dictionary = dict(zip(random_indices, self.sampled_series))
                curves = []
                for curve, _ in self.curves_indices:
                    curve_i = list(map(dictionary.get, curve[np.isin(curve, random_indices, assume_unique=True)]))
                    dictionary_i = dict(zip(curve_i, self.sampled_series))
                    curves.append([curve_i, list(map(dictionary_i.get, self.sampled_series))])

                curves = np.array(curves, dtype=np.int32)  # .astype(np.int32)  # number of curves, P

                data_glofas = self.__load_dynamic_data(files_glofas, self.variables_glofas)[:, :,
                              random_indices]  # V, T, P
                data_era5_land = self.__load_dynamic_data(files_era5_land, self.variables_era5_land)[:, :,
                                 random_indices]  # V, T, P
                data_cpc = self.__load_dynamic_data(files_cpc, self.variables_cpc)[:, :, random_indices]  # 1, Tf, P
                target_glofas = self.__load_dynamic_data(files_glofas_t, ['dis24'])[:, :, random_indices]  # 1, Tf, P
                target_obs = self.__load_dynamic_data(files_obs_t, ['dis24']).astype(
                    np.float32)[:, :, random_indices]  # 1, Tf, P

                data_hres = self.__load_hres_data(files_hres,
                                                  self.variables_hres_forecast,
                                                  np.arange(0, self.delta_t_f), forecast_type)[:, :,
                            random_indices]  # 1, Tf, P

                if self.is_obs:
                    weight_map_thr = self.generate_thr_weights_obs(target_obs, thresholds_obs)
                else:
                    weight_map_thr = self.generate_thr_weights(target_glofas, thresholds)

            else:  # get all points (full resolution)

                curves = self.curves_indices.copy()

                # get glofas variables
                data_glofas = self.__load_dynamic_data(files_glofas, self.variables_glofas)
                # get era5 land variables
                data_era5_land = self.__load_dynamic_data(files_era5_land, self.variables_era5_land).copy()
                # get cpc variables
                data_cpc = self.__load_dynamic_data(files_cpc, self.variables_cpc)
                # get static variables
                data_static = self.data_static
                # target_glofas 1, T, P
                target_glofas = self.__load_dynamic_data(files_glofas_t, ['dis24'])
                # get obs data 1, T, P
                target_obs = self.__load_dynamic_data(files_obs_t, ['dis24']).astype(np.float32)
                # data_hres V, T, P
                data_hres = self.__load_hres_data(files_hres, self.variables_hres_forecast,
                                                  np.arange(0, self.delta_t_f),
                                                  forecast_type)
                # get weights
                # data_weight = self.weight_map
                if self.is_obs:
                    thresholds_obs = self.thresholds_obs
                else:
                    thresholds = self.thresholds

                if self.is_obs:
                    weight_map_thr = self.generate_thr_weights_obs(target_obs, thresholds_obs)
                else:
                    weight_map_thr = self.generate_thr_weights(target_glofas, thresholds)

                # station_indices = np.isin(self.indices_aifas, self.indices_obs)
                random_indices = self.full_series

        # subtract from the last time step dis24
        target_glofas = target_glofas - data_glofas[self.dis24_index:self.dis24_index + 1, -1:, :]
        target_obs = target_obs - data_glofas[self.dis24_index:self.dis24_index + 1, -1:, :]

        if self.is_norm:
            # log1p transformation
            target_glofas = self.log1p_transform(target_glofas)
            target_obs = self.log1p_transform(target_obs)

            if self.variables_glofas_log1p:
                data_glofas[self._variables_glofas_log1p_indices] = self.log1p_transform(
                    data_glofas[self._variables_glofas_log1p_indices])
            if self.variables_era5_land_log1p:
                data_era5_land[self._variables_era5_land_log1p_indices] = self.log1p_transform(
                    data_era5_land[self._variables_era5_land_log1p_indices])
            if self.variables_hres_forecast_log1p:
                data_hres[self._variables_hres_forecast_log1p_indices] = self.log1p_transform(
                    data_hres[self._variables_hres_forecast_log1p_indices])
            # if self.variables_cpc_log1p:
            #    data_cpc[self._variables_cpc_log1p_indices] = self.log1p_transform(
            #        data_cpc[self._variables_cpc_log1p_indices])
            data_cpc = self.log1p_transform(data_cpc)

            data_glofas[self._variables_glofas_norm_indices] = self.transform(
                data_glofas[self._variables_glofas_norm_indices],
                self.glofas_mean[self._variables_glofas_norm_indices][:, None, None],
                self.glofas_std[self._variables_glofas_norm_indices][:, None, None])

            # for v in range(self.var_n_era5_land):
            #    if v in self._variables_era5_land_log1p_indices:
            #        continue
            #    data_era5_land[v] = self.transform(data_era5_land[v], self.era5_land_mean[v], self.era5_land_std[v])

            data_era5_land[self._variables_era5_land_norm_indices] = self.transform(
                data_era5_land[self._variables_era5_land_norm_indices],
                self.era5_land_mean[self._variables_era5_land_norm_indices][:, None, None],
                self.era5_land_std[self._variables_era5_land_norm_indices][:, None, None])

            # for v in range(self.var_n_hres):
            #    if v in self._variables_hres_forecast_log1p_indices:
            #        continue
            #    data_hres[v] = self.transform(data_hres[v], self.hres_mean[v], self.hres_std[v])

            data_hres[self._variables_hres_norm_indices] = self.transform(
                data_hres[self._variables_hres_norm_indices],
                self.hres_mean[self._variables_hres_norm_indices][:, None, None],
                self.hres_std[self._variables_hres_norm_indices][:, None, None])

        #  fill in the missing data
        data_era5_land[np.isnan(data_era5_land)] = self.nan_fill
        data_hres[np.isnan(data_hres)] = self.nan_fill

        # shuffle curves order
        if self.is_shuffle_curves:
            np.random.shuffle(curves)

        # V, T, P >> T, P, V

        return {'glofas': np.moveaxis(data_glofas, 0, -1),
                'era5': np.moveaxis(data_era5_land, 0, -1),
                'hres_forecast': np.moveaxis(data_hres, 0, -1),
                'static': data_static,
                'cpc': np.moveaxis(data_cpc, 0, -1),
                'glofas_target': np.moveaxis(target_glofas, 0, -1),
                'obs_target': np.moveaxis(target_obs, 0, -1),
                'weight': weight_map_thr,
                'curves': curves,
                # 'time': self.data_time[index],
                'file_name': self.files[index]['file_name'],
                'random_indices': random_indices,  # just for visualization in the dataset class
                'thresholds': thresholds_obs if self.is_obs else thresholds
                }

    def __len__(self):
        """ Method to get the number of samples in the dataset """
        return len(self.files)



if __name__ == '__main__':

    root_glofas_reanalysis = r'/home/ssd4tb/shams/GloFAS_Reanalysis_Global'
    root_era5_land_reanalysis = r'/home/ssd4tb/shams/ERA5-Land_Reanalysis_Global'
    root_hres_forecast = r'/home/hdd16tb/shams/paper_4/ECMWF_HRES_Global'
    root_static = r'/home/ssd4tb/shams/GloFAS_Static'
    root_cpc = r'/home/ssd4tb/shams/CPC_Global'
    root_obs = r'/home/ssd4tb/shams/GRDC_Obs_Global'

    variables_glofas = ['acc_rod24', 'dis24', 'sd', 'swi']

    """
    # HydroRIVERS
    variables_static = ['aet_mm_c01', 'aet_mm_c02', 'aet_mm_c03', 'aet_mm_c04', 'aet_mm_c05', 'aet_mm_c06',
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
                        'wet_pc_u08', 'wet_pc_u09', 'wet_pc_ug1', 'wet_pc_ug2']
    """
    # LISFLOOD
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

    variables_era5_land = ['d2m', 'e', 'es', 'evabs', 'evaow', 'evatc', 'evavt', 'lai_hv', 'lai_lv', 'pev', 'sf',
                           'skt', 'slhf', 'smlt', 'sp', 'src', 'sro', 'sshf', 'ssr', 'ssrd', 'ssro',
                           'stl1', 'stl2', 'stl3', 'stl4', 'str', 'strd', 'swvl1', 'swvl2', 'swvl3', 'swvl4',
                           't2m', 'tp', 'u10', 'v10'
                           ]

    variables_hres_forecast = ['e', 'sf', 'sp', 'ssr', 'str', 't2m', 'tp']

    variables_glofas_log1p = ['acc_rod24', 'dis24', 'sd']
    #variables_glofas_log1p = None

    """
    variables_static_log1p = ['dis_m3_pmn', 'dis_m3_pmx', 'dis_m3_pyr', 'ero_kh_cav', 'ero_kh_uav', 'gdp_ud_cav',
                              'gdp_ud_csu', 'gdp_ud_usu', 'hyd_glo_ldn', 'hyd_glo_lup', 'lkv_mc_usu', 'pop_ct_csu',
                              'pop_ct_usu', 'ppd_pk_cav', 'ppd_pk_uav', 'pre_mm_c01', 'pre_mm_c02', 'pre_mm_c03',
                              'pre_mm_c04', 'pre_mm_c05', 'pre_mm_c06', 'pre_mm_c07', 'pre_mm_c08', 'pre_mm_c09',
                              'pre_mm_c10', 'pre_mm_c11', 'pre_mm_c12', 'pre_mm_cyr', 'pre_mm_uyr', 'rdd_mk_cav',
                              'rdd_mk_uav', 'rev_mc_usu', 'rev_mc_usu', 'ria_ha_csu', 'ria_ha_usu', 'riv_tc_csu',
                              'riv_tc_usu', 'sgr_dk_rav', 'ari_ix_cav', 'ari_ix_uav', 'uparea', 'stream_pow',
                              'nli_ix_cav', 'nli_ix_uav', 'dor_pc_pva']
    """
    # variables_static_log1p = ["chanbw", "chanflpn", "elvstd", "ksat1", "ksat2", "ksat3", "soildepth2", "soildepth3",
    #                          "upArea", "waterregions"]

    variables_static_log1p = None
    variables_era5_land_log1p = None
    variables_hres_forecast_log1p = None

    years_train = [str(year) for year in range(2024, 2024 + 1)]

    dataset = RiverMamba_Dataset(
        root_glofas_reanalysis=root_glofas_reanalysis,
        root_static=root_static,
        root_era5_land_reanalysis=root_era5_land_reanalysis,
        root_hres_forecast=root_hres_forecast,
        root_cpc=root_cpc,
        root_obs=root_obs,
        nan_fill=0.,
        delta_t=2,
        delta_t_f=5,
        is_hres_forecast=True,
        is_shuffle=False,
        is_sample_aifas=True,
        is_sample=False,
        is_sample_curves=False,
        n_points=1529667,  # 86175 for EU and 1529667 for global, 6221926 for all points
        variables_glofas=variables_glofas,
        variables_era5_land=variables_era5_land,
        variables_hres_forecast=variables_hres_forecast,
        variables_static=variables_static,
        variables_cpc=['precip'],
        variables_glofas_log1p=variables_glofas_log1p,
        variables_era5_land_log1p=variables_era5_land_log1p,
        variables_hres_forecast_log1p=variables_hres_forecast_log1p,
        variables_static_log1p=variables_static_log1p,
        variables_cpc_log1p=['precip'],
        is_add_xyz=True,
        curves=['gilbert', 'gilbert_trans',
                'sweep_h',  # 'sweep_h_trans',
                'sweep_v',  # 'sweep_v_trans',
                #  'zigzag_h', 'zigzag_h_trans',
                #  'zigzag_v',# 'zigzag_v_trans',
                #'gilbert', 'gilbert_trans',
                #  'gilbert_trans'
        ],
        is_shuffle_curves=False,
        is_norm=True,
        years=years_train,
        lat_min=None,  # 30,
        lat_max=None,  # 60,
        lon_min=None,  # -10,
        lon_max=None,  # 40
        alpha=0.25,
        is_obs=False,
        static_dataset='LISFLOOD',
        is_val=True
    )

    print('number of sampled data:', dataset.__len__())
    end = time.time()
    data_temp = dataset.__getitem__(0)
    print('time: ', time.time() - end)

    print('data_glofas     shape: ', data_temp['glofas'].shape)
    print('data_era5_land  shape: ', data_temp['era5'].shape)
    print('data_hres       shape: ', data_temp['hres_forecast'].shape)
    print('data_cpc        shape: ', data_temp['cpc'].shape)
    print('data_static     shape: ', data_temp['static'].shape)
    print('target_glofas   shape: ', data_temp['glofas_target'].shape)
    print('target_obs      shape: ', data_temp['obs_target'].shape)
    print('data_weight     shape: ', data_temp['weight'].shape)
    # print('data_time       shape: ', data_temp['time'].shape)
    print('curves          shape: ', data_temp['curves'].shape)
    print('thresholds      shape: ', data_temp['thresholds'].shape)
    quit()

    #"""
    # check the curve order
    d = data_temp['glofas']
    c = data_temp['curves']
    print('points before mapping: ', d[0, :10, 0])
    dd = d[:, c[0, 0, :], :]
    print('points after mapping: ', dd[0, :10, 0])
    ddd = dd[:, c[0, 1, :], :]
    print('points after inverse mapping: ', ddd[0, :10, 0])
    #"""

    is_test_run = False
    is_test_plot = True

    if is_test_run:

        import random
        import torch

        manual_seed = 0
        random.seed(manual_seed)

        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   pin_memory=False,
                                                   num_workers=8,
                                                   # persistent_workers=False,
                                                   prefetch_factor=1
                                                   )

        end = time.time()

        from tqdm import tqdm

        for i, x in tqdm(enumerate(train_loader), total=len(train_loader)):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

    if is_test_plot:

        matplotlib.use('TkAgg')

        for i in range(len(dataset)):
            #i = np.random.choice(len(dataset), 1, replace=False)

            data_i = dataset[i]

            (data_glofas, data_era5_land, data_hres, data_cpc, data_static,
             curves, data_target, file_name, random_indices) = (data_i['glofas'],
                                                                data_i['era5'],
                                                                data_i['hres_forecast'],
                                                                data_i['cpc'],
                                                                data_i['static'],
                                                                # data_i['weight'],
                                                                # data_i['time'],
                                                                data_i['curves'],
                                                                data_i['glofas_target'],
                                                                data_i['file_name'],
                                                                data_i['random_indices']
                                                                )

            x, y = np.meshgrid(np.arange(dataset.width), np.arange(dataset.height))
            x = x.flatten()[dataset.mask_valid == 1]
            y = y.flatten()[dataset.mask_valid == 1]
            x = x[random_indices]
            y = y[random_indices]

            background_image = dataset.mask_valid.copy().astype(np.float32)
            background_image[dataset.mask_valid == 0] = np.nan
            # if random
            background_image[dataset.mask_valid == 1] = 0  # np.nan
            # background_image[dataset.ww.flatten() == 0] = np.nan
            indices = np.arange(len(background_image))[dataset.mask_valid == 1][random_indices]

            """
            # test the variables
            for v in range(dataset.var_n_cpc):
                break
                background_image[indices] = data_cpc[-1, :, v]
                plt.imshow(background_image.reshape(dataset.height, dataset.width))
                plt.colorbar()
                plt.title(dataset.variables_cpc[v])
                plt.show()
            for v in range(dataset.var_n_glofas):
                break
                background_image[indices] = data_glofas[-1, :, v]
                plt.imshow(background_image.reshape(dataset.height, dataset.width))
                plt.colorbar()
                plt.title(dataset.variables_glofas[v])
                plt.show()
            for v in range(dataset.var_n_era5_land):
                break
                background_image[indices] = data_era5_land[-1, :, v]
                plt.imshow(background_image.reshape(dataset.height, dataset.width))
                plt.colorbar()
                plt.title(dataset.variables_era5_land[v])
                plt.show()
            for v in range(dataset.var_n_hres):
                break
                background_image[indices] = data_hres[-1, :, v]
                plt.imshow(background_image.reshape(dataset.height, dataset.width))
                plt.colorbar()
                plt.title(dataset.variables_hres_forecast[v])
                plt.show()
            for v in range(dataset.var_n_static):
                background_image[indices] = data_static[:, v]
                plt.imshow(background_image.reshape(dataset.height, dataset.width))
                plt.colorbar()
                plt.show()
            #"""

            background_image[indices] = data_glofas[0, :, 1]
            background_image = background_image.reshape(dataset.height, dataset.width)
            plt.imshow(background_image)
            # plt.show()

            colors = ['bisque', 'darkcyan', 'darkcyan', 'darkcyan', 'darkcyan',
                      'bisque', 'bisque', 'bisque', 'bisque', 'bisque']

            for c in range(len(curves)):
                plt.plot(x[curves[c, 0]][:], y[curves[c, 0]][:], '.-', linewidth=1.0, c=colors[c],
                         markersize=1.0)
                break

            plt.axis('off')

            plt.show()
            plt.close()

