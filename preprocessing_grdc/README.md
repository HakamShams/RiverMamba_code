This folder includes all the scripts for processing GRDC data, applying GRDC mask, and converting time zones between utc 0 and local time.

## Code

### GRDC preprocessing

Step 0: download GRDC dataset from https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Home;

Step 1: Convert the downloaded GRDC dataset from `txt` to `csv` file.
```bash
python grdctxt2csv.py
```

Step 2: Filter out the GRDC dataset based by skiping files if catchment area is smaller than 500 km²; Fill in the observation gaps less than 7 days of dataset by linear intepolation; Convert local time of GRDC observation to UTC0; details are explained in the script.

```bash
python filter_local2utc0_grdc.py
```

The script `filter_grdc.py` only applies the first two steps without time conversion.

### Apply GRDC mask

Apply the maske of the filtered location of GRDC stations on original dataset to prepare GRDC-masked dataset for training LSTM. Using scripts in folder `./RiverMamba_code/preprocessing_grdc/apply_grdc_mask/.`


### Time conversion

The scripts in `./RiverMamba_code/preprocessing_grdc/time_convert/.` are explicitly for time conversion: `utc0tolocal.py` is to convert back utc0 to local time; `google2utc0.py` is the script we used to convert local time from google LSTM model reforecast and ungauged prediction (Nearing et al. 2024) to utc0 for our evaluation.

### Code stucture

```text
├── apply_grdc_mask
│   ├── apply_obs_mask_cpc.py
│   ├── apply_obs_mask_era5.py
│   ├── apply_obs_mask_glofas.py
│   └── apply_obs_mask_hres.py
├── filter_grdc.py
├── filter_local2utc0_grdc.py
├── grdctxt2csv.py
├── notebooks
│   ├── local2utc0_google_output.ipynb
│   ├── local2utc0_grdc_caravan.ipynb
│   ├── local2utc0_grdc.ipynb
│   ├── txt2csv.ipynb
│   └── utc02local.ipynb
└── time_convert
    ├── google2utc0.py
    └── utc0tolocal.py
```

## Reference

Nearing, G., Cohen, D., Dube, V. et al. Global prediction of extreme floods in ungauged watersheds. Nature 627, 559–563 (2024). https://doi.org/10.1038/s41586-024-07145-1.