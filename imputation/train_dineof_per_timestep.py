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
import numpy as np

import sys

sys.path.insert(0, os.getcwd())
from dataset.dataset_imputation_as_image import PRE8dDataset
from utils import check_dir, masked_mae, masked_mse, masked_cor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from utils import check_dir, masked_mae, masked_mse, seed_everything

import os
import sys

from tqdm import trange
from loguru import logger
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)


class DINEOF(BaseEstimator):
    def __init__(self, R, tensor_shape, mask=None,
                 nitemax=300, toliter=1e-5, tol=1e-8, to_center=True, 
                 keep_non_negative_only=False,
                 with_energy=False,
                 early_stopping=True):
        self.K = R  # HACK: Make interface consistent with DINEOF3, but want to keep intrinsics as is
        self.nitemax = nitemax
        self.toliter = toliter
        self.tol = tol
        self.to_center = to_center
        self.keep_non_negative_only = keep_non_negative_only
        self.tensor_shape = tensor_shape
        self.with_energy = with_energy
        self.mask = np.load(mask).astype(bool) if mask is not None else np.ones(tensor_shape).astype(bool)
        self.mask = self._broadcast_mask(self.mask, tensor_shape[-1])
        self.inverse_mask = ~self.mask
        self.early_stopping = early_stopping

    def _broadcast_mask(self, mask, t):
        mask = np.repeat(mask[:, :, None], t, axis=2)
        return rectify_tensor(mask)
        
    def score(self, X, y):
        """
            You can think of this like negative error (bigger is better due to error diminishing.
            It is made like so to be compatible with scikit-learn grid search utilities.
        """
        y_hat = self.predict(X)
        return -nrmse(y_hat, y)
    
    def rmse(self, X, y):
        return -self.score(X, y) * y.std()
    
    def nrmse(self, X, y):
        return -self.score(X, y)
        
    def predict(self):
        return self.reconstructed_tensor
        
    def fit(self, y):
        tensor = y
        self._fit(tensor)
        
    def _fit(self, mat):
        if mat.ndim > 2:
            mat = rectify_tensor(mat)

        if self.to_center:
            mat, *means = center_mat(mat)

        # Initial guess
        nan_mask = np.isnan(mat)
        non_nan_mask = ~nan_mask
        mat[nan_mask] = 0
        # Outside of an investigated area everything is considered to be zero

        conv_error = 0
        energy_per_iter = []
        for i in range(self.nitemax):
            u, s, vt = svds(mat, k=self.K, tol=self.tol)

            # Save energy characteristics for this iteration
            if self.with_energy:
                energy_i = calculate_mat_energy(mat, s)
                energy_per_iter.append(energy_i)
            
            mat_hat = u @ np.diag(s) @ vt
            mat_hat[non_nan_mask] = mat[non_nan_mask]

            new_conv_error = np.sqrt(np.mean(np.power(mat_hat[nan_mask] - mat[nan_mask], 2))) / mat[non_nan_mask].std()
            mat = mat_hat

            # pbar.set_postfix(error=new_conv_error, rel_error=abs(new_conv_error - conv_error))
            
            grad_conv_error = abs(new_conv_error - conv_error)
            conv_error = new_conv_error
            
            # logger.info(f'Error/Relative Error at iteraion {i}: {conv_error}, {grad_conv_error}')
            
            if self.early_stopping:
                break_condition = (conv_error <= self.toliter) or (grad_conv_error < self.toliter)
            else:
                break_condition = (conv_error <= self.toliter)
                
            if break_condition:              
                break

        energy_per_iter = np.array(energy_per_iter)

        if self.to_center:
            mat = decenter_mat(mat, *means)

        if self.keep_non_negative_only:
            mat[mat < 0] = 0

        # Save energies in model for distinct components (lat, lon, t)
        if self.with_energy:
            for i in range(mat.ndim):
                setattr(self, f'total_energy_{i}', np.array(energy_per_iter[:, i, 0]))
                setattr(self, f'explained_energy_{i}', np.array(energy_per_iter[:, i, 1]))
                setattr(self, f'explained_energy_ratio_{i}', np.array(energy_per_iter[:, i, 2]))

        self.final_iter = i
        self.conv_error = conv_error
        self.grad_conv_error = grad_conv_error
        self.reconstructed_tensor = mat
        self.singular_values_ = s
        self.ucomponents_ = u
        self.vtcomponents_ = vt

def unrectify_mat(mat, spatial_shape):
    tensor = []

    for t in range(mat.shape[-1]):
        col = mat[:, t]
        unrectified_col = col.reshape(spatial_shape)
        tensor.append(unrectified_col)

    tensor = np.array(tensor)
    tensor = np.moveaxis(tensor, 0, -1)

    return tensor


def rectify_tensor(tensor):
    rect_mat = []
    for t in range(tensor.shape[-1]):
        rect_mat.append(tensor[:, :, t].flatten())
    rect_mat = np.array(rect_mat)
    rect_mat = np.moveaxis(rect_mat, 0, -1)
    return rect_mat

def tensorify(X, y, shape):
    tensor = np.full(shape, np.nan)
    for i, d in enumerate(X):
        lat, lon = d.astype(np.int32)
        tensor[lat, lon] = y[i]

    return tensor

def nrmse(y_hat, y):
    """
        Normalized root mean squared error
    """
    root_meaned_sqd_diff = np.sqrt(np.mean(np.power(y_hat - y, 2)))
    return root_meaned_sqd_diff / np.std(y)

def calculate_mat_energy(mat, s):
    sample_count_0 = mat.shape[1]
    sample_coef_0 = 1 / (sample_count_0 - 1)
    total_energy_0 = np.array([sample_coef_0 * np.trace(mat @ mat.T) for _ in range(len(s))])
    expl_energy_0 = -np.sort(-sample_coef_0 * s * s)
    expl_energy_ratio_0 = expl_energy_0 / total_energy_0

    sample_count_1 = mat.shape[0]
    sample_coef_1 = 1 / (sample_count_1 - 1)
    total_energy_1 = np.array([sample_coef_1 * np.trace(mat.T @ mat) for _ in range(len(s))])
    expl_energy_1 = -np.sort(-sample_coef_1 * s * s)
    expl_energy_ratio_1 = expl_energy_1 / total_energy_1

    return np.array([[total_energy_0, expl_energy_0, expl_energy_ratio_0],
                        [total_energy_1, expl_energy_1, expl_energy_ratio_1]])
                    
def center_mat(mat):
    nan_mask = np.isnan(mat)
    temp_mat = mat.copy()
    temp_mat[nan_mask] = 0

    m0 = temp_mat.mean(axis=0)
    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] -= m0

    m1 = temp_mat.mean(axis=1)
    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] -= m1

    temp_mat[nan_mask] = np.nan
    return temp_mat, m0, m1


def decenter_mat(mat, m0, m1):
    temp_mat = mat.copy()

    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] += m0

    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] += m1

    return temp_mat

parser = argparse.ArgumentParser(description='Imputation')

# args for area and methods
parser.add_argument('--area', type=str, default='MEXICO', help='which bay area we focus')

# basic args
parser.add_argument('--epochs', type=int, default=500, help='epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
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

base_dir = "./log/imputation/{}/{}/DINEOF_per_timestep/".format(config.in_len, config.area)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
check_dir(base_dir)
seed_everything(1234)
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO, filename=os.path.join(base_dir, '{}_missing_{}.log'.format(timestamp, config.missing_ratio)), filemode='a', format='%(asctime)s - %(message)s')
print(config)
logging.info(config)

test_dloader = DataLoader(PRE8dDataset(config, mode='test'), 1, shuffle=False)

best_mae_sst = 100
best_mae_chla = 100

model = DINEOF(10, [config.height, config.width])

test_dloader_pbar = tqdm(test_dloader)
# for train_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(train_dloader_pbar):

#     tmp_data = torch.where(data_gt_masks.cpu()==0, float("nan"), datas.cpu())
#     tmp_data = torch.where(data_ob_masks.cpu()==0, float("nan"), tmp_data)
#     tmp_data = rearrange(tmp_data, "b t c h w -> (b h w c t)")
#     tmp_data = tmp_data.cpu().numpy()
#     time = torch.arange(datas.shape[1]).unsqueeze(0).unsqueeze(0).expand(datas.shape[-2], datas.shape[-1], -1).reshape(-1)
#     lati = torch.arange(datas.shape[-2]).unsqueeze(-1).unsqueeze(-1).expand(-1, datas.shape[-1], datas.shape[1]).reshape(-1)
#     lon = torch.arange(datas.shape[-1]).unsqueeze(0).unsqueeze(-1).expand(datas.shape[-2], -1, datas.shape[1]).reshape(-1)
#     x = np.stack([lati.numpy(), lon.numpy(), time.numpy()], axis=1)
#     model.fit(x, tmp_data)
 
chla_mae_list, chla_mse_list = [], []
for test_step, (datas, data_ob_masks, data_gt_masks, labels, label_masks) in enumerate(test_dloader_pbar):
    impute_data_list = []
    for t in range(datas.shape[1]):
        data = datas[:, t, :, :, :].squeeze()

        tmp_data = torch.where(data_gt_masks[:,t].cpu().squeeze()==0, float("nan"), data.cpu())
        tmp_data = torch.where(data_ob_masks[:,t].cpu().squeeze()==0, float("nan"), tmp_data)
        model.fit(tmp_data.numpy())

        imputed_data = model.predict()
        imputed_data = rearrange(imputed_data, "(b t c h) w->b t c h w", b=1, t=1, c=1, h=data.shape[-2], w=data.shape[-1])
        impute_data_list.append(torch.from_numpy(imputed_data))

    imputed_data = torch.cat(impute_data_list, dim=1)
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
