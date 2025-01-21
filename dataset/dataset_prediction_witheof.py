import warnings
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import h5py
import os
import pickle
from utils import StandardScaler, LogScaler
import copy
warnings.filterwarnings("ignore")

class PRE8dDataset(Dataset):
    def __init__(self, config, mode="train"):
        super().__init__()
        self.data_root="../data/{}/".format(config.area)
        self.in_len = config.in_len
        self.out_len = config.out_len

        self.datapath = (
            self.data_root + "/missing_" + str(config.missing_ratio) + "_in_" + str(config.in_len) + "_out_" + str(config.out_len) + "_imputed_dineof.pk"
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


        if os.path.isfile(self.datapath) is False:
            print(self.datapath + ': data is not prepared')
        else:  # load datasetfile
            with open(self.datapath, "rb") as f:
                self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = pickle.load(
                    f
                )
        self.datas = self.datas[:,:,:,self.area.astype(bool)]
        self.data_ob_masks = self.data_ob_masks[:,:,:,self.area.astype(bool)]
        self.data_gt_masks = self.data_gt_masks[:,:,:,self.area.astype(bool)]
        self.labels = self.labels[:,:,:, self.area.astype(bool)]
        self.label_ob_masks = self.label_ob_masks[:,:,:,self.area.astype(bool)]

        bound = 648 - self.in_len - self.out_len

        if mode == "train":
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[:bound], self.data_ob_masks[:bound], self.data_gt_masks[:bound], self.labels[:bound], self.label_ob_masks[:bound]
        elif mode == 'test':
            self.datas, self.data_ob_masks, self.data_gt_masks, self.labels, self.label_ob_masks = self.datas[bound:], self.data_ob_masks[bound:], self.data_gt_masks[bound:], self.labels[bound:], self.label_ob_masks[bound:]

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, index):
        return self.datas[index], self.data_ob_masks[index], self.data_gt_masks[index], self.labels[index], self.label_ob_masks[index]