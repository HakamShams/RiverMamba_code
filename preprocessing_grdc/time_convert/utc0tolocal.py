import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo


# --- Configuration ---
meta_path = "./RiverMamba_dataset/GRDC_Meta_AIFAS.txt"
input_dir = "./RiverMamba_dataset/inference_rivermamba/s_92"  # change to your actual input dir
output_dir = "./RiverMamba_dataset/inference_rivermamba/s_92_local"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Read station metadata (index_aifas, grdc_id, lat, lon)
# --------------------------
selected_data = []  # list of tuples: (index_aifas, grdc_id, lat, lon)

with open(meta_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # skip blank or comment lines

        fields = line.split(",")
        if len(fields) >= 9:
            index_aifas = fields[0].strip()
            grdc_id = fields[1].strip()
            lat = float(fields[7])
            lon = float(fields[8])
            selected_data.append((index_aifas, grdc_id, lat, lon))
        else:
            print("Skipping malformed line:", line)

print(f"Loaded {len(selected_data)} valid stations.")

# --------------------------
# Compute per-station UTC offsets (hours, rounded to int like your code)
# --------------------------
tf = TimezoneFinder()
ref_date = datetime(2020, 1, 1)   # reference date to capture DST if any
tz_offsets = {}  # map: station_index -> int hours

for i, (_, grdc_id, lat, lon) in enumerate(selected_data):
    tz_str = tf.timezone_at(lat=lat, lng=lon)
    if tz_str is None:
        tz_offsets[i] = 0
    else:
        try:
            tz = ZoneInfo(tz_str)
            offset_hours = tz.utcoffset(ref_date).total_seconds() / 3600.0
            tz_offsets[i] = int(round(offset_hours))  # keep integer-hour behavior
        except Exception as e:
            print(f"Skipping {grdc_id} at position {i}: {e}")
            tz_offsets[i] = 0

# --------------------------
# Collect input files and parse timestamps from filenames YYYYMMDD.nc
# --------------------------
nc_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nc")])
if len(nc_files) == 0:
    raise FileNotFoundError(f"No .nc files in {input_dir}")

timestamps = [np.datetime64(datetime.strptime(f.split(".")[0], "%Y%m%d"), "ns") for f in nc_files]

# --------------------------
# Read and stack all files into (timestamp, time, x)
# --------------------------
all_data = []
for f in nc_files:
    p = os.path.join(input_dir, f)
    ds = xr.open_dataset(p)
    all_data.append(ds["dis24"])   # expected shape: (time=7, x=3366)
    ds.close()

data_stack = xr.concat(all_data, dim="timestamp")          # (timestamp, time, x)
data_stack = data_stack.assign_coords(timestamp=("timestamp", timestamps))
print(f"Stacked data shape: {dict(data_stack.sizes)}")     # {'timestamp': N, 'time': 7, 'x': 3366}

# --------------------------
# Shift UTC0 daily values to local-time daily values per station
# Logic: for each station x and lead t, blend with previous/next day along 'timestamp'
# offset > 0: local(today) = (24-offset)/24 * UTC(today) + offset/24 * UTC(next)
# offset < 0: local(today) = |offset|/24 * UTC(prev) + (24-|offset|)/24 * UTC(today)
# First/last day set to NaN accordingly
# --------------------------
data_local = data_stack.copy(deep=True)

n_ts = data_stack.sizes["timestamp"]
n_time = data_stack.sizes["time"]
n_x = data_stack.sizes["x"]

# Precompute shifted arrays along timestamp
shift_next = data_stack.shift(timestamp=-1)  # UTC(next day)
shift_prev = data_stack.shift(timestamp=+1)  # UTC(prev day)

for x in range(n_x):                 # each station
    offset = tz_offsets.get(x, 0)
    if offset == 0:
        continue

    if offset > 0:
        w_today = (24 - offset) / 24.0
        w_next = offset / 24.0
        for t in range(n_time):
            q_today = data_stack[:, t, x]
            q_next = shift_next[:, t, x]
            data_local[:-1, t, x] = w_today * q_today[:-1] + w_next * q_next[:-1]
            data_local[-1, t, x] = np.nan
    else:
        abs_off = abs(offset)
        w_prev = abs_off / 24.0
        w_today = (24 - abs_off) / 24.0
        for t in range(n_time):
            q_today = data_stack[:, t, x]
            q_prev = shift_prev[:, t, x]
            data_local[1:, t, x] = w_prev * q_prev[1:] + w_today * q_today[1:]
            data_local[0, t, x] = np.nan

# --------------------------
# Save per-day files, each with variable 'dis24' and dims (time=7, x=3366)
# Filename preserved as YYYYMMDD.nc
# --------------------------
for i, ts in enumerate(data_local["timestamp"].values):
    date_str = pd.Timestamp(ts).strftime("%Y%m%d")
    one_day = data_local.isel(timestamp=i).drop_vars("timestamp")  # keep (time, x)
    ds_out = xr.Dataset({"dis24": one_day})
    out_path = os.path.join(output_dir, f"{date_str}.nc")
    ds_out.to_netcdf(out_path)

print("All daily files saved with local-time conversion (shape per file: time=7, x=3366).")