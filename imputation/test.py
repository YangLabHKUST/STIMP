import argparse
import torch
import datetime
import json
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
import sys
from calflops import calculate_flops

sys.path.insert(0, os.getcwd())
from dataset.dataset_imputation import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_cor


parser = argparse.ArgumentParser(description='Imputation')

# args for area and methods
parser.add_argument('--area', type=str, default='PRE', help='which bay area we focus')

# basic args
parser.add_argument('--epochs', type=int, default=500, help='epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--test_freq', type=int, default=500, help='test per n epochs')
parser.add_argument('--embedding_size', type=int, default=64)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--diffusion_embedding_size', type=int, default=64)
parser.add_argument('--side_channels', type=int, default=1)

# args for tasks
parser.add_argument('--in_len', type=int, default=46)
parser.add_argument('--out_len', type=int, default=46)
parser.add_argument('--missing_ratio', type=float, default=0.1)

# args for diffusion
parser.add_argument('--beta_start', type=float, default=0.0001, help='beta start from this')
parser.add_argument('--beta_end', type=float, default=0.5, help='beta end to this')
parser.add_argument('--num_steps', type=float, default=50, help='denoising steps')
parser.add_argument('--num_samples', type=int, default=10, help='n datasets')
parser.add_argument('--schedule', type=str, default='quad', help='noise schedule type')
parser.add_argument('--target_strategy', type=str, default='random', help='mask')
parser.add_argument('--method', type=str, default='CSDI', help='which method we use')

# args for mae
parser.add_argument('--num_heads', type=int, default=8, help='n heads for self attention')
config = parser.parse_args()

if config.area=="MEXICO":
    config.height, config.width = 36, 120
elif config.area=="PRE":
    config.height, config.width = 60, 96
elif config.area=="Chesapeake":
    config.height, config.width = 60, 48
elif config.area=="Yangtze":
    config.height, config.width = 96, 72
else:
    print("Not Implement")

base_dir = "./log/imputation/{}/{}/".format(config.area, config.method)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, 'memory.log'), filemode='a', format='%(asctime)s - %(message)s')
logging.info(config)

train_dataset = PRE8dDataset(config)
train_dloader = DataLoader(train_dataset, 1, shuffle=True, prefetch_factor=2, num_workers=2)
adj = np.load("data/{}/adj.npy".format(config.area))
adj = torch.from_numpy(adj).float().to(device)
low_bound = torch.from_numpy(train_dataset.min).float().to(device)
high_bound = torch.from_numpy(train_dataset.max).float().to(device)

if config.method=="CSDI":
    from model.csdi import IAP_base
    model = IAP_base(config, low_bound, high_bound)
    model = model.to(device)
    flops, macs, params = calculate_flops(model, input_shape=(1, config.in_len, 1, 4443))

elif config.method=="ImputeFormer":
    from model.imputeformer import ImputeFormer
    model = ImputeFormer(config)
    model = model.to(device)
    flops, macs, params = calculate_flops(model, input_shape=(1, config.in_len, 1, 4443))

elif config.method=="Inpainter":
    from model.inpainter import IAP_base
    model = IAP_base(config, low_bound, high_bound)
    model = model.to(device)
    flops, macs, params = calculate_flops(model, input_shape=(1, config.in_len, 1, 4443))

elif config.method=="STIMP":
    from model.graphdiffusion import IAP_base
    model = IAP_base(config, low_bound, high_bound)
    model = model.to(device)
    flops, macs, params = calculate_flops(model, input_shape=(3, config.in_len, 1, 4443))

elif config.method=="MAE":
    from model.mae import MaskedAutoEncoder
    model = MaskedAutoEncoder(config)
    model = model.to(device)
    flops, macs, params = calculate_flops(model, input_shape=(1, config.in_len, 1, 60, 96))

print("{}".format(config.method))
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))



