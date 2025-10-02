# ------------------------------------------------------------------
# Script for converting original GRDC txt files to CSV format
# 
# ------------------------------------------------------------------

import os
import glob
import re
import pandas as pd

# Set the input and output base directories
input_base = "./RiverMamba_dataset/GRDC_Caravan/GRDC"
output_base = "./RiverMamba_dataset/GRDC_Caravan/GRDC_csv"

# Create the output directory if it doesn't exist
os.makedirs(output_base, exist_ok=True)

# Function to extract header info from a file
def extract_header_info(lines):
    grdc_no, lat, lon, area = None, None, None, None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("# GRDC-No.:"):
            m = re.search(r"# GRDC-No\.:\s*(\S+)", line)
            if m:
                grdc_no = m.group(1)
        elif "# Catchment area" in line:
            m = re.search(r"# Catchment area.*:\s*(\S+)", line)
            if m:
                area = m.group(1)
        elif line.startswith("# Latitude (DD):"):
            m = re.search(r"# Latitude \(DD\):\s*(\S+)", line)
            if m:
                lat = m.group(1)
        elif line.startswith("# Longitude (DD):"):
            m = re.search(r"# Longitude \(DD\):\s*(\S+)", line)
            if m:
                lon = m.group(1)
        elif line.upper().startswith("# DATA") or line.upper() == "DATA":
            break
    
    return grdc_no, lat, lon, area

# Function to process a single txt file and write the CSV
def process_txt_file(txt_path):
    encodings = ['utf-8', 'latin-1']
    for encoding in encodings:
        try:
            with open(txt_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            print(f"Failed to decode {txt_path} with {encoding} encoding.")
            continue
    else:
        print(f"Skipping {txt_path}; unable to decode with available encodings.")
        return
    
    grdc_no, lat, lon, area = extract_header_info(lines)
    if grdc_no is None:
        print(f"Skipping {txt_path}; GRDC-No not found.")
        return
    
    data_start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("# DATA") or line.strip().upper() == "DATA":
            data_start_idx = idx + 2  # Skip the next line which is the header
            break
    
    if data_start_idx is None or data_start_idx >= len(lines):
        print(f"Skipping {txt_path}; DATA section not found.")
        return
    
    records = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 3:
            continue
        
        date_val = parts[0]
        try:
            streamflow = float(parts[2])
        except ValueError:
            streamflow = None
        
        records.append({
            "date": date_val,
            "streamflow": streamflow,
            "lat": lat,
            "lon": lon,
            "area": area
        })
    
    if records:
        df = pd.DataFrame(records)
        # Remove the first row if it contains 'YYYY-MM-DD'
        if df.iloc[0]['date'] == 'YYYY-MM-DD':
            df = df.iloc[1:]
        csv_filename = f"GRDC_{grdc_no}.csv"
        csv_path = os.path.join(output_base, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Processed '{txt_path}' -> '{csv_path}'")
    else:
        print(f"No data found in {txt_path}")

# Recursively find all txt files under the input base directory
txt_files = glob.glob(os.path.join(input_base, "**/*.txt"), recursive=True)

print(f"Found {len(txt_files)} txt files.")
for txt_file in txt_files:
    process_txt_file(txt_file)
