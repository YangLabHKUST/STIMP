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

sys.path.insert(0, os.getcwd())
from dataset.dataset_imputation import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, seed_everything

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

model = torch.load(base_dir+'best_0.1.pt')
model = model.to(device)
print(model)
logging.info(model)
datapath = "/home/mafzhang/data/{}/8d/missing_0.1_in_46_out_46_1.pk".format(config.area)
if os.path.isfile(datapath) is False:
    print("file does not exist")
    exit()
with open(datapath,'rb') as f:
    datas, data_ob_masks, data_gt_masks, labels, label_ob_masks = pickle.load(
                    f
                )

adj = np.load("/home/mafzhang/data/MEXICO/8d/adj.npy")
is_sea = np.load("/home/mafzhang/data/MEXICO/8d/is_sea.npy").astype(bool)
adj = torch.from_numpy(adj).float().to(device)


bs = 24
step = datas.shape[0]//bs + 1
num_samples = 10

imputed_datas=[]
for i in tqdm(range(step)):
    data = datas[bs*i:min(bs*i+bs, datas.shape[0])]
    data_mask = data_ob_masks[bs*i:min(bs*i+bs, datas.shape[0])]


    data_graph = torch.from_numpy(data[:,:,:,is_sea]).float().to(device)
    data_mask_graph = torch.from_numpy(data_mask[:,:,:,is_sea]).to(device)

    imputed_data = model.impute(data_graph, data_mask_graph, adj, node_type, 10)
    data_mask_graph = data_mask_graph.unsqueeze(1).expand_as(imputed_data)
    data_graph = data_graph.unsqueeze(1).expand_as(imputed_data)
    imputed_data = data_mask_graph.cpu()*data_graph.cpu() + (1-data_mask_graph.cpu())*imputed_data
    imputed_datas.append(imputed_data)

imputed_datas_graph = torch.cat(imputed_datas,dim=0)
imputed_datas = torch.zeros(imputed_datas_graph.shape[0],10,46,1,is_sea.shape[0],is_sea.shape[1])
imputed_datas[:,:,:,:,is_sea]=imputed_datas_graph
new_data_path="/home/mafzhang/data/{}/8d/missing_0.1_in_46_out_46_1_imputed_graph.pk".format(config.area)
with open(new_data_path, 'wb') as f:
    pickle.dump([imputed_datas.numpy(), data_ob_masks,data_gt_masks,labels,label_ob_masks], f)
