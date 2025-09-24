# ------------------------------------------------------------------
# Simple script to visualize the GRDC observational river discharge data
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

root_cpc = r'/home/ssd4tb/shams/GRDC_Obs_Global/'
root_static = r'/home/ssd4tb/shams/GloFAS_Static/'

# define the variables to be visualized
variable = 'dis24'

# image size
height, width = 3000, 7200

# ------------------------------------------------------------------

# get the mask on land from static data and prepare an empty image
mask_valid = xr.open_dataset(os.path.join(root_static, "masks/mask_valid.nc"))
mask_valid = mask_valid['mask_valid'].values.flatten().astype(np.float32)
mask_valid[mask_valid == 0] = np.nan

# ------------------------------------------------------------------

# read the available years inside the grdc folder
years = os.listdir(root_cpc)
years.sort()

years = [year for year in years if not year.endswith('.nc') and not year.endswith('.json')]

# visualize each year separately
for year in years:

    # directory for the year
    dir_year = os.path.join(root_cpc, year)
    # read files inside the year directory
    files = os.listdir(dir_year)
    files.sort()

    files = [file for file in files if file.endswith('.nc')]

    # visualize each file for each year separately
    for file in files:
        # directory for the file
        dir_file = os.path.join(dir_year, file)

        # read the netcdf data
        data = xr.open_dataset(dir_file)

        data_v = data[variable].values

        background_image = mask_valid.copy()
        background_image[background_image == 1] = data_v

        background_image = background_image.reshape(height, width)

        idx_data = ~np.isnan(background_image)
        idx = np.argwhere(idx_data)

        # check if there is data to vis
        if len(idx) == 0:
            continue

        plt.scatter(idx[:, 1], idx[:, 0], c=background_image[idx_data], s=10)
        plt.colorbar()

        plt.imshow(background_image)
        plt.title('variable=' + variable + ', file=' + file[:-3])
        plt.show()
        plt.close()

