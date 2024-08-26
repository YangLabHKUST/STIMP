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
from dataset.dataset_prediction import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_huber_loss, huber_loss, mse_loss, seed_everything
from model.cross_models.cross_former import Crossformer
from torchtsmixer import TSMixer
from model.mtgnn import MTGNN
from model.graphtransformer import GraphTransformer
import numpy as np
from utils import AverageMeter

parser = argparse.ArgumentParser(description='Prediction')

# args for area and methods
parser.add_argument('--area', type=str, default='MEXICO', help='which bay area we focus')
parser.add_argument('--method', type=str, default='GraphTransformer', help='which bay area we focus')
parser.add_argument('--index', type=int, default=0, help='which dataset we use')

# basic args
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
parser.add_argument('--test_freq', type=int, default=20, help='test per n epochs')
parser.add_argument('--hidden_dim', type=int, default=8)

# args for tasks
parser.add_argument('--in_len', type=int, default=46)
parser.add_argument('--out_len', type=int, default=46)
parser.add_argument('--missing_ratio', type=float, default=0.1)

config = parser.parse_args()

if config.area=="MEXICO":
    config.height, config.width = 36, 120
elif args.area=="PRE":
    config.height, config.width = 60, 96
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
adj = np.load("/home/mafzhang/data/{}/8d/adj.npy".format(config.area))
adj = torch.from_numpy(adj).float().to(device)
is_sea = torch.from_numpy(train_dataset.area).bool().to(device)
low_bound = torch.from_numpy(train_dataset.min).float().to(device)
high_bound = torch.from_numpy(train_dataset.max).float().to(device)
mean = torch.from_numpy(train_dataset.mean).float().to(device)
std = torch.from_numpy(train_dataset.std).float().to(device)
   
assert config.method in ["GraphTransformer", "MTGNN"], print("{} not implement".format(config.method))
if config.method == "GraphTransformer":
    model = GraphTransformer(config, adj, is_sea, mean, std)
elif config.method == "MTGNN":
    model = MTGNN(adj=adj, gcn_true=True, build_adj=False, kernel_set=[7,7], kernel_size=7, gcn_depth=1, num_nodes=adj.shape[0], dropout=0.3, subgraph_size=20, node_dim=8,dilation_exponential=2, conv_channels=2, residual_channels=2, skip_channels=4, end_channels=8, seq_length=config.in_len, in_dim=1, out_dim=config.out_len, layers=2, propalpha=0.5, tanhalpha=3, layer_norm_affline=False)

model = model.to(device)
train_process = tqdm(range(config.epochs))
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
optimizer_scheduler = CosineAnnealingLR(optimizer, config.epochs)

best_mae_chla = 100

for epoch in train_process:
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    optimizer_scheduler.step(epoch)
    end = time.time()
    for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
        datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
        datas = datas*data_ob_masks
        means = datas.mean(1, keepdim=True).detach()
        datas = datas - means
        stdev = torch.sqrt(torch.var(datas, dim=1, keepdim=True, unbiased=False) + 1e-5)
        datas/= stdev
        prediction = model(datas)
        prediction = prediction*stdev + means
        loss = masked_mse(prediction, labels, label_masks)

        loss.backward()
        optimizer.step()

        losses_m.update(loss.item(), datas.size(0))
        data_time_m.update(time.time() - end)
        torch.cuda.synchronize()

    log_buffer = "train prediction loss with dataset {}: {:.4f}".format(losses_m.avg, config.index)
    log_buffer += "| time : {:.4f}".format(data_time_m.avg)
    train_process.write(log_buffer)
    losses_m.reset()
    data_time_m.reset()
    if epoch % config.test_freq == 0 and epoch != 0:
        predictions = []
        labels_list = []
        label_masks_list = []
        datas_list = []
        data_masks_list = []
        with torch.no_grad():
            for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
                datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
                datas = datas*data_ob_masks
                means = datas.mean(1, keepdim=True).detach()
                datas = datas - means
                stdev = torch.sqrt(torch.var(datas, dim=1, keepdim=True, unbiased=False) + 1e-5)
                datas /=stdev

                prediction = model(datas)
                prediction = prediction*stdev + means
                mask = label_masks.cpu()
                label = labels.cpu()

                predictions.append(prediction[:,:,0].cpu())
                labels_list.append(label[:,:,0])
                label_masks_list.append(mask[:,:,0])

        predictions = torch.cat(predictions, 0)
        labels = torch.cat(labels_list, 0)
        label_masks = torch.cat(label_masks_list, 0)
        chla_mae = (torch.abs(predictions - labels) * label_masks).sum([1]) / (label_masks.sum([1]) + 1e-5)
        chla_mse = (((predictions - labels) * label_masks) ** 2).sum([1]) / (label_masks.sum([1]) + 1e-5)
        chla_mae = chla_mae.mean(0)
        chla_mse = chla_mse.mean(0)
        chla_mae_mean = chla_mae.mean()
        chla_mse_mean = chla_mse.mean()

        log_buffer = "MAE: all area mean is {:.4f}, MSE: all area mean is {:.4f}\t".format(chla_mae_mean, chla_mse_mean)

        train_process.write(log_buffer)
        logging.info(log_buffer)

        if chla_mae_mean < best_mae_chla:
            torch.save(model, base_dir + "best_model_with_dataset_{}.pt".format(config.method, config.index))
            np.save(base_dir + "prediction_{}.npy".format(str(config.index)), predictions)
            best_mae_chla = chla_mae_mean
