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


```bash
bash run.py --gene_map YOUR_PATH/gene_map.tif \  # The gene map is a 3D image with shape (height, width, n_genes)
--nuclei_mask YOUR_PATH/nuclei_mask.tif \      # The nuclei mask is a 2D image with shape (height, width)
--log_dir ./log/LOG_NAME
```
