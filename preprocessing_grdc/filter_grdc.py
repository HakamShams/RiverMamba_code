# Only filter the stations but without time conversion

import os
import glob
import pandas as pd
import datetime
import numpy as np
from timezonefinder import TimezoneFinder
import pytz

# -----------------------------------------------------------
# Step 1. Read metadata and catchment area information
# -----------------------------------------------------------
# Read gauge coordinate metadata from all CSV files in GRDC_csv directory
input_dir = './RiverMamba_dataset/GRDC_Caravan/GRDC_csv'
output_dir = './RiverMamba_dataset/GRDC_Caravan/GRDC_csv_filtered'
os.makedirs(output_dir, exist_ok=True)

gauge_coords = {}
gauge_area = {}

for file_path in glob.glob(os.path.join(input_dir, '*.csv')):
    gauge_id = os.path.basename(file_path).split('.')[0]
    df = pd.read_csv(file_path, nrows=3)  # Read only the first row for metadata
    if df.empty or len(df) == 1:
        print(f"File {file_path} is empty or has only the header. Skipping.")
        continue
    lat, lon, area = df['lat'].values[0], df['lon'].values[0], df['area'].values[0]
    gauge_coords[gauge_id] = (lat, lon)
    gauge_area[gauge_id] = area


# -----------------------------------------------------------
# Step 3. convert units from mm/day to m3/s.
#
#   (a) We load the gauge CSV (with dates and streamflow in mm/day) -- NOT NEED TO DO THIS FOR GRDC original dataset
#   (b) We reindex the data to a complete daily time series.
#   (c) We detect gaps. For gaps shorter than 7 days, we fill missing days
#       via linear interpolation; for longer gaps, we treat the data as separate segments;
#       ### if the total missing gap days exceed 1500, we skip the CSV file.
#   (d) We remove catchment areas smaller than 500 km².
#
#       (The first or last day of each segment is skipped accordingly.)
#
# -----------------------------------------------------------


os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the input directory.


for file_path in glob.glob(os.path.join(input_dir, '*.csv')):
    gauge_id = os.path.basename(file_path).split('.')[0]
    print(f"\nProcessing gauge {gauge_id} ...")
    
    # Load the time series data with date parsing.
    df = pd.read_csv(file_path, parse_dates=['date'])

    df.sort_values('date', inplace=True)
    # Filter to only include data from January 1, 1979 onward.
    df = df[df['date'] >= pd.Timestamp('1979-01-01')]

    # Filter out rows with streamflow value equal to -999
    df = df[df['streamflow'] != -999]

    if df.empty or len(df) == 1:  # Skip empty files or files with only the header
        print(f"File {file_path} is empty or has only the header after 1979. Skipping.")
        continue

    df.reset_index(drop=True, inplace=True)
    
    # Ensure we have the 'streamflow' column.
    if 'streamflow' not in df.columns:
        print(f"File {file_path} does not have a 'streamflow' column. Skipping.")
        continue
    
    # Set the date as index.
    df.set_index('date', inplace=True)
    
    # Create a complete daily date range spanning the observed period.
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_full = df.reindex(full_index)
    # Mark which dates are originally observed.
    df_full['observed'] = ~df_full['streamflow'].isna()
    
    # -------------------------------------------------------
    # Identify continuous segments.
    # We want to group days where the gap between consecutive observed dates is less than 7 days.
    # -------------------------------------------------------
    observed_dates = df_full.index[df_full['observed']]
    if len(observed_dates) == 0:
        print(f"No observed dates for {gauge_id}. Skipping file.")
        continue
    total_gap_days = 0
    gap_count = 0
    segments = []
    seg_start = observed_dates[0]
    prev_date = observed_dates[0]
    
    for current_date in observed_dates[1:]:
        gap = (current_date - prev_date).days
        # Count missing days: if gap > 1 then missing days = gap - 1
        if gap > 1:
            total_gap_days += (gap - 1)
        if gap < 8:  # if gap is less than 7 days, consider it continuous
            prev_date = current_date
        else:
            print(f"Gap detected: {gap} days between {prev_date.strftime('%Y-%m-%d')} and {current_date.strftime('%Y-%m-%d')}")
            gap_count += 1
            segments.append((seg_start, prev_date))
            seg_start = current_date
            prev_date = current_date

    # Append the last segment.
    segments.append((seg_start, prev_date))
    print(f"Total gaps detected in {gauge_id}: {gap_count}")
    print(f"Total missing gap days in {gauge_id}: {total_gap_days} days")
    
    # If total missing gap days exceed 1500, skip this CSV file.
    if total_gap_days > 1500:
        print(f"Skipping file {gauge_id} because total missing gap days ({total_gap_days}) exceed 10% days between 1979 to 2022.")
        continue

    # -------------------------------------------------------
    # Skip file if catchment area is smaller than 500 km².
    # make it comparable with our 5km resolution model output.
    # -------------------------------------------------------
    area = gauge_area.get(gauge_id)
    if area is None:
        print(f"Catchment area not found for {gauge_id}. Skipping unit conversion.")
        continue
    if area < 500:
        print(f"Skipping file {gauge_id} because catchment area ({area} km²) is smaller than 500 km².")
        continue
    
    # For each segment, reindex with daily frequency and fill gaps by linear interpolation.
    converted_segments = []
    for seg_start, seg_end in segments:
        seg_index = pd.date_range(start=seg_start, end=seg_end, freq='D')
        seg_df = df_full.loc[seg_start:seg_end].reindex(seg_index)
        # Only fill gaps if they exist; if the gap is small (<7 days) this will interpolate.
        seg_df['streamflow'] = seg_df['streamflow'].interpolate(method='linear')
        seg_df = seg_df.copy()  # work on a copy
        
        # Keep only the computed Q_utc and the date index.
        converted_segments.append(seg_df[['streamflow']])
    
    if not converted_segments:
        print(f"No valid segments found for {gauge_id}.")
        continue

    # Combine all segments into one DataFrame (they remain separate in time).
    df_conv = pd.concat(converted_segments).sort_index()

    df_conv['streamflow'] = df_conv['streamflow']

    # Round the streamflow values to 3 decimal places
    df_conv['streamflow'] = df_conv['streamflow'].round(3)
    
    # Prepare the output DataFrame (date and converted streamflow in m3/s).
    df_out = df_conv[['streamflow']].reset_index().rename(columns={'index': 'date'})
    # -------------------------------------------------------
    # Append gauge metadata: latitude, longitude, and catchment area.
    # -------------------------------------------------------
    if gauge_id in gauge_coords:
        lat, lon = gauge_coords[gauge_id]
    else:
        lat, lon = np.nan, np.nan
    df_out['lat'] = lat
    df_out['lon'] = lon
    df_out['area'] = area
    
    # Save to a new CSV file (same gauge id as file name).
    output_file = os.path.join(output_dir, f"{gauge_id}.csv")
    df_out.to_csv(output_file, index=False)
    print(f"Processed {gauge_id}: converted data saved to {output_file}")

print("\nAll files have been processed.")