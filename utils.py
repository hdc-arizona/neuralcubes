import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LRS

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DENetDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = np.loadtxt(txt_file, dtype=np.float32)
        self.transform = transform
        # self.min_count,self.max_count = np.min(self.data[:,-1]),np.max(self.data[:,-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #normalized_label = 2*(self.data[idx][-1:]-self.min_count)/(self.max_count-self.min_count) - 1
        normalized_label = self.data[idx][-1:]
        sample = {'vec': self.data[idx][0:-1],
                  'label': normalized_label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class DENetDatasetWeighted(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = np.loadtxt(txt_file, dtype=np.float32)
        self.transform = transform
        # self.min_count,self.max_count = np.min(self.data[:,-1]),np.max(self.data[:,-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #normalized_label = 2*(self.data[idx][-1:]-self.min_count)/(self.max_count-self.min_count) - 1
        normalized_label = self.data[idx][-1:]
        sample = {'vec': self.data[idx][0:-2],
                  'weight': self.data[idx][-2:-1],
                  'label': normalized_label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class DENetRangesDataset(Dataset):
    def __init__(self, txt_file, transform=None, schema=None):
        self.data = np.loadtxt(txt_file, dtype=float)
        # print(self.data.shape)
        self.transform = transform

        self.data_schema = schema['data_schema']
        self.net_branches = schema['net_schema']['branches']

        self.dims = []
        for branch in self.net_branches:
            if self.data_schema[branch['key']]['type'] != 'spatial':
                d = self.data_schema[branch['key']]['dimension']
                self.dims.append(d)
            else:
                spatial_res =self.data_schema[branch['key']]['resolution']
                self.dims += [spatial_res['x'], spatial_res['y']]
        self.rangeDims = len(self.dims)*2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'ranges': self.data[idx][:self.rangeDims],
                  'counts': self.data[idx][self.rangeDims:]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class LogCounts(object):
    def __call__(self, sample):
        vec, label = sample['vec'], sample['label']
        label = np.log1p(label)
        return {'vec': vec, 'label':label}


class ToTensor(object):
    def __init__(self, dtype="float"):
        self.dtype = dtype

    def __call__(self, sample):
        vec, label = sample['vec'], sample['label']
        if self.dtype == "float":
            tensor = {}
            for key in sample:
                tensor[key] = torch.from_numpy(sample[key]).float()
            return tensor
        elif self.dtype == "double":
            tensor = {}
            for key in sample:
                tensor[key] = torch.from_numpy(sample[key]).double()
            return tensor
        else:
            return sample
