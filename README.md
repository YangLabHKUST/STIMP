# AI-powered spatiotemporal imputation and prediction of chlorophyll-a concentration in coastal oceans
This repository contains the code for the STIMP method, an advanced AI framework to impute and predict Chl_a across a broad spatiotemporal scale in coastal oceans. STIMP's results can be utilized to diagnose and analyze the ecosystem health of coastal oceans based on the remote sensing measurement.


<img src="https://github.com/user-attachments/assets/47b87208-e49a-45e0-9c93-8d792546bcac"  width="1000" />


## Installation
You can install the development version of STIMP:
```bash
git clone https://github.com/YangLabHKUST/STIMP.git
cd /path/to/STIMP
conda create -n stimp python=3.9
conda activate stimp
pip install -r requirements.txt
```

## Four representative coastal ocean area analysis
The code for reproducing the results presented in our paper are available on the [This tutorial](https://stimp-tutorials.readthedocs.io/). To reproduce our resluts, it is necessary to first train STIMP and the baselines, which can be found in the tutorials:
* Train STIMP on each coastal ocean area(https://stimp-tutorials.readthedocs.io/en/latest/usage.html)
* Train baselines, including imputation methods and prediction methods on each coastal ocean area(https://stimp-tutorials.readthedocs.io/en/latest/baselines.html)

The resluts presented in our paper are available:
* [Pearl River Estuary](https://stimp-tutorials.readthedocs.io/en/latest/analysis/PRE/index.html)
* [Northern Gulf of Mexico](https://stimp-tutorials.readthedocs.io/en/latest/analysis/MEXICO/index.html)
* [Chesapeake Bay](https://stimp-tutorials.readthedocs.io/en/latest/analysis/Chesapeake/index.html)
* [Yangtze River Estuary](https://stimp-tutorials.readthedocs.io/en/latest/analysis/Yangtze/index.html)



