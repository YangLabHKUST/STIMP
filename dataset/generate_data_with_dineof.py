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
from einops import rearrange

sys.path.insert(0, os.getcwd())
from dataset.dataset_imputation import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, seed_everything
from model.dineof import DINEOF
import pickle

parser = argparse.ArgumentParser(description='Imputation')

# args for area and methods
parser.add_argument('--area', type=str, default='MEXICO', help='which bay area we focus')

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

base_dir = "./log/imputation/{}/GraphDiffusion/".format(config.area)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
seed_everything(1234)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}_missing_{}.log'.format(timestamp, config.missing_ratio)), filemode='a', format='%(asctime)s - %(message)s')
# print(config)
# logging.info(config)

datapath = "/home/mafzhang/data/{}/8d/missing_0.1_in_46_out_46_1.pk".format(config.area)
if os.path.isfile(datapath) is False:
    print("file does not exist")
    exit()
with open(datapath,'rb') as f:
    datas, data_ob_masks, data_gt_masks, labels, label_ob_masks = pickle.load(
                    f
                )

adj = np.load("/home/mafzhang/data/{}/8d/adj.npy".format(config.area))
is_sea = np.load("/home/mafzhang/data/{}/8d/is_sea.npy".format(config.area)).astype(bool)
adj = torch.from_numpy(adj).float().to(device)
model = DINEOF(10, [config.height, config.width, config.in_len])

bs = 1
step = datas.shape[0]//bs + 1
num_samples = 10

imputed_datas=[]
for i in tqdm(range(step)):
    data = torch.from_numpy(datas[bs*i:min(bs*i+bs, datas.shape[0])])
    data_mask = torch.from_numpy(data_ob_masks[bs*i:min(bs*i+bs, datas.shape[0])])
    tmp_data = torch.where(data_mask.cpu()==0, float("nan"), data)
    tmp_data = rearrange(tmp_data, "b t c h w -> (b h w c t)")
    tmp_data = tmp_data.cpu().numpy()
    time = torch.arange(datas.shape[1]).unsqueeze(0).unsqueeze(0).expand(datas.shape[-2], datas.shape[-1], -1).reshape(-1)
    lati = torch.arange(datas.shape[-2]).unsqueeze(-1).unsqueeze(-1).expand(-1, datas.shape[-1], datas.shape[1]).reshape(-1)
    lon = torch.arange(datas.shape[-1]).unsqueeze(0).unsqueeze(-1).expand(datas.shape[-2], -1, datas.shape[1]).reshape(-1)
    x = np.stack([lati.numpy(), lon.numpy(), time.numpy()], axis=1)
    model.fit(x, tmp_data)

    imputed_data = model.predict(x)
    imputed_data = rearrange(imputed_data, "(b t c h w)->b t c h w", b=1, t=datas.shape[1], c=1, h=datas.shape[-2], w=datas.shape[-1])

    imputed_data = data.cpu().numpy()*data_mask.cpu().numpy() + (1-data_mask.cpu().numpy())*imputed_data
    imputed_datas.append(imputed_data)

imputed_datas_graph = np.concatenate(imputed_datas,axis==0)
new_data_path="/home/mafzhang/data/{}/8d/missing_0.1_in_46_out_46_1_imputed_dineof.pk".format(config.area)
with open(new_data_path, 'wb') as f:
    pickle.dump([imputed_datas.numpy(), data_ob_masks,data_gt_masks,labels,label_ob_masks], f)
