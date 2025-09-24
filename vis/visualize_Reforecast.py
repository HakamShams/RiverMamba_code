# ------------------------------------------------------------------
# Simple script to visualize the RiverMamba reforecast at high-resolution
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

root_reforecast = r'/home/ssd4tb/shams/Reforecasts/RiverMamba/RiverMamba_glofas_reanalysis_full_map/'
root_static = r'/home/ssd4tb/shams/GloFAS_Static/'

# define the variables to be visualized
variable = 'dis24'

# image size
height, width = 3000, 7200

# forecast step from 0 to 6
step = 0

# whether to use log1p transformation for the visualization
is_log1p = True

# ------------------------------------------------------------------

# get the mask on land from static data and prepare an empty image
mask_valid = xr.open_dataset(os.path.join(root_static, "masks/mask_valid.nc"))
mask_valid = mask_valid['mask_valid'].values.flatten().astype(np.float32)
mask_valid[mask_valid == 0] = np.nan

# ------------------------------------------------------------------

# read files inside the directory
files = os.listdir(root_reforecast)
files.sort()

files = [file for file in files if file.endswith('.nc')]

# visualize each file at step 1

for file in files:
    # directory for the file
    dir_file = os.path.join(root_reforecast, file)

    # read the netcdf data
    data = xr.open_dataset(dir_file)

    data_v = data[variable].isel(time=step).values

    if is_log1p:
        data_v = np.log1p(data_v)

    background_image = mask_valid.copy()
    background_image[background_image == 1] = data_v

    plt.imshow(background_image.reshape(height, width))
    plt.title('variable=' + variable + ', file=' + file[:-3])
    plt.colorbar()
    plt.show()
    plt.close()

