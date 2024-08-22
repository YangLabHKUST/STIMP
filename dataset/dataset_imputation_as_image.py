import warnings
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import h5py
import os
import pickle
import copy
warnings.filterwarnings("ignore")

class PRE8dDataset(Dataset):
    def __init__(self, config, mode="train"):
        super().__init__()
        self.data_root="/home/mafzhang/data/{}/8d/".format(config.area)
        self.in_len = config.in_len
        self.out_len = config.out_len

        self.datapath = (
            self.data_root + "/missing_" + str(config.missing_ratio) + "_in_" + str(config.in_len) + "_out_" + str(config.out_len) + "_1.pk"
        )
        self.mode = mode
        self.adj = np.load(self.data_root+"adj.npy")
        self.area = np.load(self.data_root+"is_sea.npy")
        mean = np.load(self.data_root+"mean.npy")
        std = np.load(self.data_root+"std.npy")
        self.mean = mean[self.area.astype(bool)]
        self.std = std[self.area.astype(bool)]
        max = np.load(self.data_root+"max.npy")
        min = np.load(self.data_root+"min.npy")
        self.max = max[self.area.astype(bool)]
        self.min = min[self.area.astype(bool)]

        oral_data, oral_mask = self.load_data()

        if os.path.isfile(self.datapath) is False:
            datas, data_ob_masks, data_gt_masks, labels, label_ob_masks = [], [], [], [], []
            for index in range(len(oral_data) - self.in_len - self.out_len):
                data = oral_data[index:index+self.in_len]
                data_ob_mask = oral_mask[index:index+self.in_len]
                label = oral_data[index+self.in_len:index+self.in_len+self.out_len]
                label_ob_mask = oral_mask[index+self.in_len:index+self.in_len+self.out_len]

                masks = data_ob_mask.reshape(-1).copy()
                obs_indices = np.where(masks)[0].tolist()
                miss_indices = np.random.choice(
                    obs_indices, (int)(len(obs_indices) * config.missing_ratio), replace=False
                )
                masks[miss_indices] = False
                gt_masks = masks.reshape(data_ob_mask.shape)
                datas.append(data)
                data_ob_masks.append(data_ob_mask)
                data_gt_masks.append(gt_masks)
                labels.append(label)
                label_ob_masks.append(label_ob_mask)

            self.datas = np.array(datas).astype("float32")
            self.data_ob_masks = np.array(data_ob_masks).astype("float32")
            self.data_gt_masks = np.array(data_gt_masks).astype("float32")
            self.labels = np.array(labels).astype("float32")
            self.label_ob_masks = np.array(label_ob_masks).astype("float32")
            with open(self.datapath, "wb") as f:
                pickle.dump(
                    [self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks], f
                )
        else:  # load datasetfile
            with open(self.datapath, "rb") as f:
                self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = pickle.load(
                    f
                )

        # self.datas = self.datas[:,:,:,self.area.astype(bool)]
        # self.data_ob_masks = self.data_ob_masks[:,:,:,self.area.astype(bool)]
        # self.data_gt_masks = self.data_gt_masks[:,:,:,self.area.astype(bool)]
        # self.labels = self.labels[:,:,:,self.area.astype(bool)]
        # self.label_ob_masks = self.label_ob_masks[:,:,:,self.area.astype(bool)]
        bound = 648 - self.in_len - self.out_len
        if mode == "train":
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[:bound], self.data_ob_masks[:bound], self.data_gt_masks[:bound], self.labels[:bound], self.label_ob_masks[:bound]
        elif mode == 'test':
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[bound:], self.data_ob_masks[bound:], self.data_gt_masks[bound:], self.labels[bound:], self.label_ob_masks[bound:]


    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        return self.datas[index], self.data_ob_masks[index], self.data_gt_masks[index], self.labels[index], self.label_ob_masks[index]

    def load_data(self):
        chla = np.load(self.data_root+"/chla.npy")
        chla = chla[:,np.newaxis]
        chla_mask = ~np.isnan(chla)
        self.chla_scaler = LogScaler()
        chla = self.chla_scaler.transform(chla)
        chla = np.nan_to_num(chla, nan=0.)

        mask = chla_mask.astype(np.float32)
        return chla, mask

class LogScaler:
    def transform(self, x):
        return np.log10(x)
    def inverse_transform(self, x):
        return 10**x
if __name__ == "__main__":
    dataset = PRE8dDataset(data_root='/home/mafzhang/data/PRE/8d', in_len=12, out_len=1,missing_ratio=0.1, mode='train')

