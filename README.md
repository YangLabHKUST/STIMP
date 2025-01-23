# AI-powered spatiotemporal imputation and prediction of chlorophyll-a concentration in coastal oceans
This repository contains the code for the STIMP method, an advanced AI framework to impute and predict Chl_a across a broad spatiotemporal scale in coastal oceans. STIMP's results can be utilized to diagnose and analyze the ecosystem health of coastal oceans based on the remote sensing measurement.


<img src="https://github.com/user-attachments/assets/47b87208-e49a-45e0-9c93-8d792546bcac"  width="1000" />

## Reproducibility
We provide [source code](https://github.com/Ryanfzhang/STIMP/tree/release/tutorials) for reproducing the experiments of the paper "AI-powered spatiotemporal imputation and prediction of chlorophyll-a concentration in coastal oceans".

### Step1. Install
```bash
git clone https://github.com/YangLabHKUST/STIMP.git
cd /path/to/STIMP
conda create -n stimp python=3.9
conda activate stimp
pip install -r requirements.txt
```

### Step2. Prepare data
All data used in this work are publicly available through online sources. The chlorophyll-a observation datasets were 8-day averaged Level 3 mapped products from Moderate Resolution Imaging Spectroradiometer (MODIS) Aqua projects with a spatial resolution of 4 km https://search.earthdata.nasa.gov/search?q=10.5067/AQUA/MODIS/L3M/CHL/2022. You can select the data with ***.8D.*.4km.nc** as filter. 

We also uploaded the datasets on Zenodo at https://doi.org/10.5281/zenodo.14724760. Then, 
```bash
mv data.zip /path/to/STIMP/
unzip e data.zip
```
[Prepare the dataset from the raw data](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/01-preprocess_chla_data.ipynb) We generate the 4 datasets, including Pearl River Estuary, the Northern of Mexico, Chesapeake Bay and Yangtze River Estuary, following this tutorials. The generated datasets are also included in the data.zip

### Step3. Train the imputation function $p_\theta$ of STIMP

Taking the Pearl River Estuary as an example, we construct 9 datasets with different missing rate. We train the STIMP with each dataset:
```bash
for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
  python imputation/train_stimp.py --missing_ratio $i --area PRE
done
```
Baselines can be trained:
```bash
for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
  python imputation/train_cf.py --missing_ratio $i --area PRE
  python imputation/train_csdi.py --missing_ratio $i --area PRE
  python imputation/train_dineof.py --missing_ratio $i --area PRE
  python imputation/train_imputeformer.py --missing_ratio $i --area PRE
  python imputation/train_inpainter.py --missing_ratio $i --area PRE
  python imputation/train_lin_itp.py --missing_ratio $i --area PRE
  python imputation/train_mae.py --missing_ratio $i --area PRE
  python imputation/train_mean.py --missing_ratio $i --area PRE
  python imputation/train_trmf.py --missing_ratio $i --area PRE
done
```
Some visualization results are contained within [Imputation in Pearl River Estuary](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/02-imputation-pearl-river-estuary.ipynb)

For other coastal ocean areas, STIMP and baselines are trained by replacing PRE with MEXICO, Chesapeake or Yangtze.
+ [Imputation in the Northern of MEXICO](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/05-imputation-mexico.ipynb)
+ [Impuation in Chesapeake Bay](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/08-imputation-chesapeake-bay.ipynb)
+ [Imputation in Yangtze River Estuary](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/11-imputation-yangtze-estuary.ipynb)

### Step4. Impute the observation
Observations of Chl_a in Pearl River Estuary are imputed:
```bash
python dataset/generate_data_with_stimp.py --area PRE
```

### Step5. Train the prediction fuction $p_\Phi$ of STIMP
We sample 10 different imputed Chl_a distribution from $p_\theta(\mathbf{X}|\mathbf{X}^{ob})$. Then we can learn 10 different $p_\Phi(\tilde{\mathbf{Y}}|\mathbf{X})$ due to differet input $\mathbf{X}$:
```bash
for i in {0..9}  
do  
  python prediction/train.py --index $i --area PRE
done
```
Baselines are learned based on the original observations $\mathbf{X}^{ob}$:
```bash
python prediction/train_without_spatial_imputation.py --method "CrossFormer" --area PRE
python prediction/train_without_spatial_imputation.py --method "iTransformer" --area PRE
python prediction/train_without_spatial_imputation.py --method "TSMixer" --area PRE
python prediction/train_without_imputation.py --method "MTGNN" --area PRE
python prediction/train_as_image_without_imputation.py --method "PredRNN" --area PRE
python prediction/train_xgboost_without_imputation.py --area PRE
```

We also train baselines based on the imputed Chl_a distribution (in supplementary material):
```bash
for i in {0..9}  
do  
  python prediction/train_without_spatial.py --method "CrossFormer" --area PRE --index $i
  python prediction/train_without_spatial.py --method "iTransformer" --area PRE --index $i
  python prediction/train_without_spatial.py --method "TSMixer" --area PRE --index $i
  python prediction/train.py --method "MTGNN" --area PRE --index $i
  python prediction/train_as_image.py --method "PredRNN" --area PRE --index $i
  python prediction/train_xgboost.py --area PRE --index $i
done
```

We provide the source code for overall prediction performance in each coastal ocean area:
+ [Overall prediction performance in Pearl River Estuary](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/03-prediction-pearl-river-estuary.ipynb)
+ [Overall prediction performance in the Northern of MEXICO](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/06-prediction-mexico.ipynb)
+ [Overall prediction performance in Chesapeake Bay](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/09-prediction-chesapeake-bay.ipynb)
+ [Overall prediction performance in Yangtze River Estuary](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/12-prediction-yangtze-estuary.ipynb)

Some case studies are included in the following tutorials:
+ [Case study in Pearl River Estuary](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/04-prediction-pearl-river-estuary-case-study.ipynb)
+ [Case study in the Northern of MEXICO](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/07-prediction-mexico-case-study.ipynb)
+ [Case study in Chesapeake Bay](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/10-prediction-chesapeake-bay-case-study.ipynb)
+ [Case study in Yangtze River Estuary](https://github.com/YangLabHKUST/STIMP/blob/release/tutorials/13-prediction-yangtze-estuary-case-study.ipynb)
