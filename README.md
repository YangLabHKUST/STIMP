# AI-powered spatiotemporal imputation and prediction of chlorophyll-a concentration in coastal oceans
This repository contains the code for the STIMP method, an advanced AI framework to impute and predict Chl_a across a broad spatiotemporal scale in coastal oceans. STIMP's results can be utilized to diagnose and analyze the ecosystem health of coastal oceans based on the remote sensing measurement.


<img src="https://github.com/user-attachments/assets/47b87208-e49a-45e0-9c93-8d792546bcac"  width="1000" />

## Reproducibility
We provide [source code](https://github.com/Ryanfzhang/STIMP/tree/release/tutorials) for reproducing the experiments of the paper "AI-powered spatiotemporal imputation and prediction of chlorophyll-a concentration in coastal oceans".

## Installation
```bash
git clone https://github.com/YangLabHKUST/STIMP.git
cd /path/to/STIMP
conda create -n stimp python=3.9
conda activate stimp
pip install -r requirements.txt
```
## Pipeline
See the `script` dir for examples on how to run STIMP and other baselines.

### Imputation
If we want to run STIMP for Chl_a imputation in the Pearl River Estuary, we can run the following script

```bash
bash script/train_stimp_imputation.sh PRE # Chesapeake, MEXICO and Yangtze for the other three costal oceans 
```
### Prediction
We can make robust prediction of Chl_a across a large scale coastal ocean after imputation.

```bash
bash script/train_graphtransformer.sh PRE # Chesapeake, MEXICO and Yangtze for the other three costal oceans 
```
## Data
All data used in this work are publicly available through online sources. The chlorophyll-a observation datasets were 8-day averaged Level 3 mapped products from Moderate Resolution Imaging Spectroradiometer (MODIS) Aqua projects with a spatial resolution of 4 km https://search.earthdata.nasa.gov/search?q=10.5067/AQUA/MODIS/L3M/CHL/2022. We also uploaded the datasets on Zenodo at https://doi.org/10.5281/zenodo.14638406. 
