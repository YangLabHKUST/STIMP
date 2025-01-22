import torch
import yaml
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

sys.path.insert(0, os.getcwd())
from xgboost import XGBRegressor
from dataset.dataset_prediction import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss, huber_loss, mse_loss, seed_everything
import numpy as np
from utils import AverageMeter

parser = argparse.ArgumentParser(description='Prediction')

# args for area and methods
parser.add_argument('--area', type=str, default='MEXICO', help='which bay area we focus')
parser.add_argument('--method', type=str, default='XGBoost', help='which bay area we focus')
parser.add_argument('--index', type=int, default=0, help='which dataset we use')

# basic args
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--test_freq', type=int, default=20, help='test per n epochs')
parser.add_argument('--hidden_dim', type=int, default=8)

# args for tasks
parser.add_argument('--in_len', type=int, default=46)
parser.add_argument('--out_len', type=int, default=46)
parser.add_argument('--missing_ratio', type=float, default=0.1)

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

flag = "without_imputation"
base_dir = "./log/prediction/{}/{}/{}/".format(config.area, config.method, flag)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
seed_everything(1234)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}_dataset_{}.log'.format(timestamp, config.index)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)

train_dataset = PRE8dDataset(config)
train_dloader = DataLoader(train_dataset, config.batch_size, shuffle=True, prefetch_factor=2, num_workers=2) 
test_dloader = DataLoader(PRE8dDataset(config, mode='test'), config.batch_size, shuffle=False)
adj = np.load("./data/{}/adj.npy".format(config.area))
adj = torch.from_numpy(adj).float().to(device)
is_sea = torch.from_numpy(train_dataset.area).bool().to(device)
low_bound = torch.from_numpy(train_dataset.min).float().to(device)
high_bound = torch.from_numpy(train_dataset.max).float().to(device)
mean = train_dataset.mean[np.newaxis, np.newaxis, np.newaxis,:]
std = train_dataset.std[np.newaxis, np.newaxis, np.newaxis,:]
   
model = XGBRegressor()
train_datas = []
train_labels = []
for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
    datas[~data_ob_masks.bool()] = np.nan
    datas = datas - mean
    labels = labels - mean
    # datas = datas*data_ob_masks

    datas = torch.permute(datas, (0, 3, 1, 2)).numpy()
    labels = torch.permute(labels, (0, 3, 1, 2)).numpy()
    labels = labels[:, :, :,  0]
    masks = torch.permute(data_ob_masks, (0, 3, 1, 2))
    datas = datas.reshape(-1, config.in_len * 1)
    labels = labels.reshape(-1, config.out_len * 1)
    labels = np.nan_to_num(labels, nan=0.)
    train_datas.append(datas)
    train_labels.append(labels)

train_datas = np.concatenate(train_datas, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
print("For dataset, Train start")
model.fit(train_datas, train_labels)
print("For dataset, Test start")

predictions = []
labels_list = []
label_masks_list = []
with torch.no_grad():
    for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
        b, t, c, n = datas.shape
        datas[~data_ob_masks.bool()] = np.nan
        datas = datas - mean
        # datas = datas*data_ob_masks

        datas = torch.permute(datas*data_ob_masks, (0, 3, 1, 2))
        datas = datas.reshape(-1, t * c)
        prediction = model.predict(datas)
        prediction = prediction.reshape(b, n, 1, -1)
        prediction = np.moveaxis(prediction, [1, -1], [-1, 1])  # b t c n
        prediction = prediction + mean
        prediction = torch.from_numpy(prediction)
        mask = label_masks.cpu()
        label = labels.cpu()

        predictions.append(prediction.cpu())
        labels_list.append(label)
        label_masks_list.append(mask)

    predictions = torch.cat(predictions, 0)
    labels = torch.cat(labels_list, 0)
    label_masks = torch.cat(label_masks_list, 0)
    chla_mae = (torch.abs(predictions - labels) * label_masks).sum([1, 2]) / (label_masks.sum([1, 2]) + 1e-5)
    chla_mse = (((predictions - labels) * label_masks) ** 2).sum([1, 2]) / (label_masks.sum([1, 2]) + 1e-5)
    chla_mae = chla_mae.mean(0)
    chla_mse = chla_mse.mean(0)

    chla_mae_mean = chla_mae[chla_mae!=0].mean()
    chla_mse_mean = chla_mse[chla_mse!=0].mean()

    log_buffer = "For dataset , test mae - {:.4f}, ".format(chla_mae_mean)
    log_buffer += "test mse - {:.4f}".format(chla_mse_mean)

    print(log_buffer)
    logging.info(log_buffer)
    np.save(base_dir + "prediction_{}.npy".format(config.index), predictions)
