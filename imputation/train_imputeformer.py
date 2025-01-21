import torch
import os
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm
from timm.utils import AverageMeter
import numpy as np
import argparse
import sys

sys.path.insert(0, os.getcwd())
from dataset.dataset_imputation import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, seed_everything
from model.imputeformer import ImputeFormer

parser = argparse.ArgumentParser(description='Imputation')

# args for area and methods
parser.add_argument('--area', type=str, default='MEXICO', help='which bay area we focus')

# basic args
parser.add_argument('--epochs', type=int, default=500, help='epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--test_freq', type=int, default=500, help='test per n epochs')
parser.add_argument('--embedding_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--diffusion_embedding_size', type=int, default=32)
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

base_dir = "./log/imputation/{}/ImputeFormer/".format(config.area)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
seed_everything(1234)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}_missing_{}.log'.format(timestamp, config.missing_ratio)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)

train_dataset = PRE8dDataset(config)
train_dloader = DataLoader(train_dataset, config.batch_size, shuffle=True, prefetch_factor=2, num_workers=2)
test_dloader = DataLoader(PRE8dDataset(config, mode='test'), config.batch_size, shuffle=False)
adj = np.load("data/{}/adj.npy".format(config.area))
adj = torch.from_numpy(adj).float().to(device)
low_bound = torch.from_numpy(train_dataset.min).float().to(device)
high_bound = torch.from_numpy(train_dataset.max).float().to(device)

model = ImputeFormer(config)
model = model.to(device)
train_process = tqdm(range(1, config.epochs + 1))
optimizer = torch.optim.Adam(model.parameters(), config.lr, weight_decay=config.wd)

p1 = int(0.75 * config.epochs)
p2 = int(0.9 * config.epochs)
optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)


best_mae_sst = 100
best_mae_chla = 100
for epoch in train_process:
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    optimizer_scheduler.step(epoch)
    end = time.time()
    for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader):
        datas, data_ob_masks, data_gt_masks, labels, label_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device), labels.to(device), label_masks.to(device)
        loss = model.trainstep(datas, data_ob_masks)
        losses_m.update(loss.item(), datas.size(0))
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        torch.cuda.synchronize()

    log_buffer = "train loss : {:.4f}".format(losses_m.avg)
    log_buffer += "| time : {:.4f}".format(data_time_m.avg)
    end = time.time()
    train_process.set_description(log_buffer)
    train_process.write(log_buffer)

    if epoch % config.test_freq == 0 and epoch != 0:
        chla_mae_list, chla_mse_list = [], []
        imputed_data_list = []
        for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader):
            datas, data_ob_masks, data_gt_masks = datas.float().to(device), data_ob_masks.to(device), data_gt_masks.to(device)

            imputed_data = model.impute(datas, data_gt_masks)

            mask = (data_ob_masks - data_gt_masks).detach().cpu()
            chla_mae = masked_mae(imputed_data[:, :, 0].detach().cpu(), datas[:, :, 0].cpu(), mask[:, :, 0])
            chla_mse = masked_mse(imputed_data[:, :, 0].detach().cpu(), datas[:, :, 0].cpu(), mask[:, :, 0])
            chla_mae_list.append(chla_mae)
            chla_mse_list.append(chla_mse)

        chla_mae = torch.stack(chla_mae_list, 0)
        chla_mse = torch.stack(chla_mse_list, 0)
        chla_mae = chla_mae[chla_mae != 0].mean()
        chla_mse = chla_mse[chla_mse != 0].mean()

        log_buffer = "test mae: chla-{:.4f}, ".format(chla_mae)
        log_buffer += "test mse: chla-{:.4f}".format(chla_mse)
        print(log_buffer)
        logging.info(log_buffer)
        if chla_mae < best_mae_chla:
            torch.save(model, base_dir+'best_{}.pt'.format(config.missing_ratio))
