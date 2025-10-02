import os
import pandas as pd
import datetime
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo  # Replaces pytz

import os
import xarray as xr
import numpy as np
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import datetime

def apply_utc_shift(ds, offset):
    """Shift daily values from local time to UTC using weighted average."""
    ds = ds.copy()

    # Step 0: Correct time axis (shift back 1 day since timestamps are right-labeled)
    ds['time'] = ds['time'] - np.timedelta64(1, 'D')

    # Step 1: Apply UTC shift based on offset
    if offset == 0:
        ds['streamflow_utc'] = ds['google_prediction']
    elif offset > 0:
        w1 = (24 - offset) / 24
        w2 = offset / 24
        shifted = ds['google_prediction'].shift(time=-1)
        ds['streamflow_utc'] = w1 * ds['google_prediction'] + w2 * shifted
        ds = ds.isel(time=slice(0, -1))  # drop last time step
    elif offset < 0:
        abs_offset = abs(offset)
        w1 = abs_offset / 24
        w2 = (24 - abs_offset) / 24
        shifted = ds['google_prediction'].shift(time=1)
        ds['streamflow_utc'] = w1 * shifted + w2 * ds['google_prediction']
        ds = ds.isel(time=slice(1, None))  # drop first time step

    return ds

def convert_units(ds, area_km2):
    """Convert from mm/day to m³/s and rename to final variable name."""
    ds = ds.copy()
    var_name = 'google_prediction_m3s_utc0'
    ds[var_name] = (ds['streamflow_utc'] * area_km2) / 86.4
    ds[var_name].attrs['units'] = 'm3/s'
    ds[var_name].attrs['description'] = 'Google forecast streamflow converted to m³/s and aligned to UTC'
    return ds.drop_vars('streamflow_utc')

def process_netcdf_file(filepath, offset, area_km2, output_dir):
    """Full processing pipeline for one NetCDF file."""
    ds = xr.open_dataset(filepath)

    # Apply conversion steps
    ds = apply_utc_shift(ds, offset)
    ds = convert_units(ds, area_km2)

    # Save to new NetCDF file
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)
    ds.to_netcdf(output_path)
    print(f"Processed and saved: {output_path}")


# -----------------------------------------------------------
# Step 1. Read metadata and filter by gauge IDs present in .nc files
# -----------------------------------------------------------
# Path configuration
attributes_file = './RiverMamba_dataset/GRDC_Caravan/GRDC_Caravan_extension_csv/attributes/grdc/GRDC_Stations.csv'
nc_dir = './RiverMamba_dataset/inference_google/kfold_splits_google_filtered'

# Get gauge IDs from the .nc files (remove prefix and suffix)
gauge_ids_to_keep = set(os.path.splitext(f)[0].replace("GRDC_", "") 
                        for f in os.listdir(nc_dir) if f.endswith(".nc"))

print(f"Found {len(gauge_ids_to_keep)} gauge IDs from .nc files.")

# Read the attributes CSV and filter to only those gauge IDs
df_attributes = pd.read_csv(attributes_file, encoding='latin1')
df_attributes['grdc_no'] = df_attributes['grdc_no'].astype(str)
df_attributes = df_attributes[df_attributes['grdc_no'].isin(gauge_ids_to_keep)]

print(f"Retained {len(df_attributes)} gauge IDs with matching metadata.")

# Build mapping dictionary: gauge_id -> (lat, lon)
gauge_coords = {row['grdc_no']: (row['lat'], row['long']) for _, row in df_attributes.iterrows()}

# -----------------------------------------------------------
# Step 2. Determine each gauge's UTC time zone offset (in hours)
# -----------------------------------------------------------
tf = TimezoneFinder()
gauge_tz_offset = {}  # Store UTC offset for each gauge
rep_date = datetime.datetime(2020, 1, 1)  # Representative date to get timezone offset

print("Calculating timezone offsets ...\n")

for gauge_id, (lat, lon) in gauge_coords.items():
    tz_str = tf.timezone_at(lng=lon, lat=lat)
    if tz_str is None:
        print(f"Timezone not found for {gauge_id} (lat={lat}, lon={lon}). Using UTC (offset = 0).")
        gauge_tz_offset[gauge_id] = 0
    else:
        try:
            tz = ZoneInfo(tz_str)
            localized = rep_date.replace(tzinfo=tz)
            offset_hours = localized.utcoffset().total_seconds() / 3600
            gauge_tz_offset[gauge_id] = offset_hours
            print(f"Gauge {gauge_id}: Timezone {tz_str}, Offset {offset_hours} hours")
        except Exception as e:
            print(f"Failed to process timezone for {gauge_id} ({tz_str}): {e}. Using UTC (offset = 0).")
            gauge_tz_offset[gauge_id] = 0

print("Timezone offset calculation completed.")

# -----------------------------------------------------------
# Step 3. Create gauge_id -> area_km2 mapping
# -----------------------------------------------------------
# 1. Extract gauge IDs from .nc filenames
gauge_ids_in_folder = set(
    os.path.splitext(f)[0].replace("GRDC_", "") 
    for f in os.listdir(nc_dir) if f.endswith(".nc")
)

# 2. Load CSV and match only those gauges
df_attributes = pd.read_csv(attributes_file, encoding='latin1')
df_attributes['grdc_no'] = df_attributes['grdc_no'].astype(str)

df_matched = df_attributes[df_attributes['grdc_no'].isin(gauge_ids_in_folder)]

# 3. Create dictionary: gauge_id → area_km2
gauge_area = {
    row['grdc_no']: row['area']
    for _, row in df_matched.iterrows()
}

print(f"Created gauge_area for {len(gauge_area)} gauges.")

input_dir = './RiverMamba_dataset/inference_google/kfold_splits_google_filtered'
output_dir = './RiverMamba_dataset/inference_google/kfold_splits_google_filtered_converted'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith('.nc'):
        gauge_id = file.replace("GRDC_", "").replace(".nc", "")
        area = gauge_area.get(gauge_id)
        offset = gauge_tz_offset.get(gauge_id)
        if area is None or offset is None:
            print(f"Skipping {gauge_id}: missing area or offset")
            continue
        filepath = os.path.join(input_dir, file)
        process_netcdf_file(filepath, offset, area, output_dir)