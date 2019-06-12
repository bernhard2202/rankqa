#!/usr/bin/env python3
"""  Proper torch data loading """

import random

import torch
from torch.utils.data.dataset import Dataset

from .data_utils import vectorize


class PairwiseRankingDataSet(Dataset):

    def __init__(self, subsampled, normalizers):
        # build vectors
        self.Xa = []
        self.Xb = []
        self.y = []
        for xa, xb in subsampled:
            if random.randint(1, 2) == 1:
                self.Xa.append((vectorize(xa, normalizers)))
                self.Xb.append((vectorize(xb, normalizers)))
                self.y.append(torch.tensor(float(xa['target'])))
            else:
                self.Xa.append((vectorize(xb, normalizers)))
                self.Xb.append((vectorize(xa, normalizers)))
                self.y.append(torch.tensor(float(xb['target'])))
        self.num_feat = len(self.Xa[0])

    def __getitem__(self, index):
        return self.Xa[index], self.Xb[index], self.y[index]

    def __len__(self):
        return len(self.y)


class StreamedDataSet(Dataset):

    def __init__(self, subsampled, normalizers):
        # build vectors
        self.Xa = []
        self.y = []
        for xa, xb in subsampled:
            self.Xa.append((vectorize(xa, normalizers)))
            self.Xa.append((vectorize(xb, normalizers)))
            self.y.append(torch.tensor(float(xa['target'])))
            self.y.append(torch.tensor(float(xb['target'])))
        self.num_feat = len(self.Xa[0])

    def __getitem__(self, index):
        return self.Xa[index], self.y[index]

    def __len__(self):
        return len(self.y)
