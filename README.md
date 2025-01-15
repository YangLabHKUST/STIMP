# AI-powered spatiotemporal imputation and prediction of chlorophyll-a concentration in coastal oceans
This repository contains the code for the STIMP method, an advanced AI framework to impute and predict Chl_a across a broad spatiotemporal scale in coastal oceans. STIMP's results can be utilized to diagnose and analyze the ecosystem health of coastal oceans based on the remote sensing measurement.


<img src="https://github.com/user-attachments/assets/47b87208-e49a-45e0-9c93-8d792546bcac"  width="1000" />

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
bash script/train_stimp_imputation.sh PRE
```
