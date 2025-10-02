#!/bin/sh

# you can download the data directly from [https://doi.org/10.60507/FK2/T8QYWE]

mkdir -p RiverMamba_pretrained_models

# RiverMamba_aifas_grdc_obs
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13848 -O RiverMamba_pretrained_models/RiverMamba_aifas_grdc_obs.pth
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13849 -O RiverMamba_pretrained_models/RiverMamba_aifas_grdc_obs.txt

# RiverMamba_aifas_reanalysis
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13850 -O RiverMamba_pretrained_models/RiverMamba_aifas_reanalysis.pth
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13851 -O RiverMamba_pretrained_models/RiverMamba_aifas_reanalysis.txt

# RiverMamba_full_map_reanalysis
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13852 -O RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.pth
wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/13853 -O RiverMamba_pretrained_models/RiverMamba_full_map_reanalysis.txt

