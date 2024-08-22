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
from einops import repeat, rearrange
import torchcde

from dataset.dataset import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_cor
from model.diffusion import IAP_base
from model.mae import MaskedAutoEncoder
from model.cf import NCF
from model.trmf import trmf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


with open("./config/config_mae.yaml", 'r') as f:
    config = yaml.safe_load(f)

base_dir = "./log/mice/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(base_dir, '{}.log'.format(timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)


train_dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d', in_len=config['in_len'],
                        out_len=config['out_len'],missing_ratio=config['missing_ratio'],
                         mode='train')
chla_scaler = train_dataset.chla_scaler
train_dloader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
test_dloader = DataLoader(PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d', in_len=config['in_len'], 
                        out_len=config['out_len'],missing_ratio=config['missing_ratio'],
                         mode='test'), 1, shuffle=False)


best_mae_sst = 100
best_mae_chla = 100
imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True)

for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
    datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

    tmp_data = torch.where(data_gt_masks.cpu()==0, float("nan"), datas.cpu())
    tmp_data = rearrange(tmp_data, "b t c h w -> (b h w c) t")
    imp.fit(tmp_data)

 
chla_mae_list, chla_mse_list = [], []
for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
    datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)

    tmp_data = torch.where(data_gt_masks.cpu()==0, float("nan"), datas.cpu())
    tmp_data = rearrange(tmp_data, "b t c h w -> (b h w c) t")
    itp_data = imp.transform(tmp_data)

    imputed_data = rearrange(torch.from_numpy(itp_data), "(b h w c) t -> b t c h w", t=datas.shape[1], b=datas.shape[0], h=datas.shape[3], w=datas.shape[4])

    mask = (data_ob_masks - data_gt_masks).cpu()
    chla_mae= masked_mae(imputed_data[:,:,0], datas[:,:,0].cpu(), mask[:,:,0])
    chla_mse= masked_mse(imputed_data[:,:,0], datas[:,:,0].cpu(), mask[:,:,0])
    chla_mae_list.append(chla_mae)
    chla_mse_list.append(chla_mse)

chla_mae = torch.stack(chla_mae_list, 0)
chla_mse = torch.stack(chla_mse_list, 0)
chla_mae = chla_mae[chla_mae!=0].mean()
chla_mse = chla_mse[chla_mse!=0].mean()

log_buffer = "test mae: chla-{:.4f}, ".format(chla_mae)
log_buffer += "test mse: chla-{:.4f}".format(chla_mse)
print(log_buffer)
logging.info(log_buffer)
