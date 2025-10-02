import xarray as xr
import numpy as np
import pandas as pd
import os
import sys

# Get the year from command-line arguments
year = int(sys.argv[1])

# Define the base directories
base_dir = "./RiverMamba_dataset/GloFAS_Reanalysis"
output_base_dir = "./RiverMamba_dataset/GRDC_masked_dataset/GloFAS_Reanalysis"

mask_obs = xr.open_dataset("./RiverMamba_dataset/GloFAS_Static/masks/mask_GRDC_obs_points.nc")
nan_stations = './RiverMamba_dataset/script/nan_indices_era5obsmask.npy'
nan_stations_indices = np.load(nan_stations)
# Loop over each year from 1979 to 2024
year_dir = os.path.join(base_dir, str(year))
output_year_dir = os.path.join(output_base_dir, str(year))

# Create the output directory if it doesn't exist
os.makedirs(output_year_dir, exist_ok=True)

# Generate all dates for the year
dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')

# Loop over each date
for date in dates:
    date_str = date.strftime("%Y%m%d")
    file_name = f"{date_str}.nc"
    file_path = os.path.join(year_dir, file_name)
    output_file_path = os.path.join(output_year_dir, file_name)
    
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Open the dataset
        glofas = xr.open_dataset(file_path)
        
        # List the data variables
        data_variables = list(glofas.data_vars)
        
        # Dictionary to hold the data arrays
        data_dict = {}
        
        # Loop through each variable name and extract the values based on the mask
        for var in data_variables:
            print (var)
            test = glofas[var].values[mask_obs['mask_points'].values == 1]
            test_cleaned = np.delete(test, nan_stations_indices[0])
            data_dict[var] = (['x'], test_cleaned)
        
        # Create a new xarray Dataset
        ds = xr.Dataset(data_dict)
        
        # Save the Dataset to a new NetCDF file
        ds.to_netcdf(output_file_path)

        print (date)
