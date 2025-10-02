#!/bin/sh

# you can download the data directly from [https://doi.org/10.60507/FK2/T8QYWE]

mkdir -p Reforecasts

# meta data
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13833 -O Reforecasts/GRDC_Meta.txt

# RiverMamba reforecasts
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13858 -O Reforecasts/RiverMamba_glofas_reanalysis.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13859 -O Reforecasts/RiverMamba_grdc_obs.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13856 -O Reforecasts/RiverMamba_glofas_reanalysis_full_map.7z

# LSTM reforecasts
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13860 -O Reforecasts/LSTM_glofas_reanalysis.7z
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13857 -O Reforecasts/LSTM_grdc_obs.7z

