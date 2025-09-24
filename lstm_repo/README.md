The directory contains all necessary scripts to train the baseline LSTM model. The LSTM model has the same structure (Encoder-decoder LSTM/Bi-LSTM) as Nearing, G., Cohen, D., Dube, V. et al. Global prediction of extreme floods in ungauged watersheds. Nature 627, 559–563 (2024). https://doi.org/10.1038/s41586-024-07145-1. As the codes in the Nature paper are not open-sourced, we followed the code from https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/modelzoo/handoff_forecast_lstm.py developed by the same group to build up our LSTM baseline model. Other than the model structure, we have our new training and evaluation schemes in corresponding to the RiverMamba model. Note: we are not training LSTM in an ungaued setup.

## Setup

Please load the env before training the LSTM:

```bash
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```

## Code

### Configuration

The main config folder includes the config files for both training and inference. The Bi-LSTM uses a weighted loss function based on the return period of the flood event. The LSTM verion uses the normal loss function. In our setup, the training used 4 Nvidia A100 GPUs. The models are trained in two stages.

Step 1: pretrain the model using ERA5 and GloFAS reanalysis dataset; During pretraining, a mask was created from the locations of 3,366 GRDC stations worldwide, and the models were trained only on the corresponding points in the reanalysis dataset defined by this mask. The trained models are saved under `./RiverMamba_code/lstm_repo/log`.
```bash
python ./RiverMamba_code/lstm_repo/train_lstm/train_lstm_era5_pretrain.py
python ./RiverMamba_code/lstm_repo/train_lstm/train_bilstm_era5_pretrain_weightloss.py
```

Step 2: fine-tune the model using HRES forecast and GRDC observations. Similary, the fine-tuning is also conducted on the location of 3366 GRDC stations globally. The trained models are saved under `./RiverMamba_code/lstm_repo/log`.

```bash
python ./RiverMamba_code/lstm_repo/train_lstm/train_lstm_eobs_finetune_pretrain.py
python ./RiverMamba_code/lstm_repo/train_lstm/train_bilstm_obs_finetune_weightloss.py
```

Inference (the path to save the inference should be defined in the config files):

```bash
python ./RiverMamba_code/lstm_repo/infer_lstm/infer_lstm.py
python ./RiverMamba_code/lstm_repo/infer_lstm/infer_bilstm.py
```

### Code stucture

```text
├── config_lstm
│   ├── config_edlstm_era5_bilstm.py
│   ├── config_edlstm_era5.py
│   ├── config_edlstm_infer_bilstm.py
│   ├── config_edlstm_infer.py
│   ├── config_edlstm_obs_bilstm.py
│   └── config_edlstm_obs.py
├── dataset
│   ├── LSTM_dataset.py
│   └── LSTM_weightloss_dataset.py
├── infer_lstm
│   ├── inference_bilstm.py
│   └── inference_lstm.py
├── models_lstm
│   ├── build_bilstm_weightloss.py
│   ├── build_lstm.py
│   ├── fc.py
│   ├── handoff_bilstm.py
│   ├── handoff_lstm.py
│   ├── head
│   │   ├── head_lstm.py
│   │   └── MLP.py
│   ├── head_lstm.py
│   ├── __init__.py
│   ├── inputlayer_lstm.py
│   ├── losses.py
│   └── positional_encoding.py
├── train_lstm
│   ├── train_bilstm_era5_pretrain_weightloss.py
│   ├── train_bilstm_obs_finetune_weightloss.py
│   ├── train_lstm_era5_pretrain.py
│   └── train_lstm_obs_finetune.py
└── utils
    └── utils_edlstm.py\
└── log
```

## Reference

Nearing, G., Cohen, D., Dube, V. et al. Global prediction of extreme floods in ungauged watersheds. Nature 627, 559–563 (2024). https://doi.org/10.1038/s41586-024-07145-1.


Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for Deep Learning research in hydrology. Journal of Open Source Software, 7(71), 4050. https://doi.org/10.21105/joss.04050


Yikui Zhang, Silvan Ragettli, Peter Molnar, Olga Fink, Nadav Peleg, Generalization of an Encoder-Decoder LSTM model for flood prediction in ungauged catchments, Journal of Hydrology, Volume 614, Part B, 2022, 128577, ISSN 0022-1694, https://doi.org/10.1016/j.jhydrol.2022.128577.