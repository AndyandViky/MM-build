# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: proposess.py
@time: 2020/9/17 10:56
@desc: proposess.py
'''
import os
import pandas as pd
import numpy as np
import scipy.io as scio
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from config import DATASETS_DIR
from utils import filter_process


def preprocess():

    name_head = 'char'
    train_names = ['01(B)', '02(D)', '03(G)', '04(L)', '05(O)', '06(Q)', '07(S)', '08(V)', '09(Z)', '10(4)', '11(7)', '12(9)']
    for i in range(2, 3):
        single_train_datas = []
        single_test_datas = []
        single_train_labels = []
        single_test_labels = []
        for j in range(12):
            train_datas = pd.read_excel('{}/S{}/S{}_train_data.xlsx'.format(DATASETS_DIR, i, i), sheet_name=name_head + train_names[j], header=None).values
            train_labels = pd.read_excel('{}/S{}/S{}_train_event.xlsx'.format(DATASETS_DIR, i, i), sheet_name=name_head + train_names[j], header=None).values

            train_datas = filter_process(train_datas.T).T
            # draw useful data
            train_labels = train_labels[1:]
            train_labels = np.delete(train_labels, np.where(train_labels[:, 0] == 100)[0], axis=0)
            begin = train_labels[:, 1] - 1 - 55
            temp_datas = []
            for s in range(150):
                temp_datas.append(train_datas[begin+s])
            single_train_datas.append(np.array(temp_datas).transpose((1, 2, 0)))
            single_train_labels.append(train_labels[:, 0])

        for j in range(9):
            test_datas = pd.read_excel('{}/S{}/S{}_test_data.xlsx'.format(DATASETS_DIR, i, i), sheet_name='char{}'.format(j+13), header=None).values
            test_labels = pd.read_excel('{}/S{}/S{}_test_event.xlsx'.format(DATASETS_DIR, i, i), sheet_name='char{}'.format(j+13), header=None).values

            test_datas = filter_process(test_datas.T).T
            # draw useful data
            test_labels = test_labels[1:]
            test_labels = np.delete(test_labels, np.where(test_labels[:, 0] == 100)[0], axis=0)
            begin = test_labels[:, 1] - 1 - 55
            temp_datas = []
            for s in range(150):
                temp_datas.append(test_datas[begin + s])
            single_test_datas.append(np.array(temp_datas).transpose((1, 2, 0)))
            single_test_labels.append(test_labels[:, 0])

        single_train_datas = np.array(single_train_datas)
        single_test_datas = np.array(single_test_datas)
        single_train_labels = np.array(single_train_labels)
        single_test_labels = np.array(single_test_labels)
        scio.savemat('{}/S{}/postprocess.mat'.format(DATASETS_DIR, i),
                     {'train': single_train_datas, 'test': single_test_datas,
                      'train_labels': single_train_labels, 'test_labels': single_test_labels})
# preprocess()


class P300(Dataset):

    def __init__(self, root, train=True, transform=None, full=False, extend=False, datas=None, labels=None, test=False):
        super(P300, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform

        self.data, self.label = self.load_p300_data(self.root, self.train, full, extend, test)

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        if self.transform is not None:
            data = self.transform(data)

        return data.type(torch.float32), label

    def __len__(self):

        return len(self.label)

    def change_labels(self, Y: np.ndarray) -> np.ndarray:

        char_map = {
            'B': (1, 8),
            'D': (1, 10),
            'G': (2, 7),
            'L': (2, 12),
            'O': (3, 9),
            'Q': (3, 11),
            'S': (4, 7),
            'V': (4, 10),
            'Z': (5, 8),
            '4': (5, 12),
            '7': (6, 9),
            '9': (6, 11),
        }

        map_values = list(char_map.values())
        for i, item in enumerate(Y):
            first = map_values[i][0]
            second = map_values[i][1]
            Y[i][Y[i] == first] = 1
            Y[i][Y[i] == second] = 1
            Y[i][Y[i] != 1] = 0
        return Y

    def load_p300_data(self, root, train=True, full=False, extend=False, test=False):

        channel_s1 = np.array([20, 17, 16, 19, 14, 11, 9, 13, 7, 18]) - 1
        channel_s2 = np.array([4, 6, 13, 15, 17, 18, 20, 19, 8, 12]) - 1
        channel_s3 = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]) - 1
        channel_s4 = np.array([1, 2, 3, 9, 10, 12, 18, 19, 20, 7, 4, 5]) - 1
        channel_s5 = np.array([4, 5, 6, 14, 16, 11, 9, 17, 20, 19, 15]) - 1

        data = scio.loadmat(os.path.join(root, 'postprocess.mat'))
        if test:
            X = data['test'].reshape((-1, data['test'].shape[2], data['test'].shape[3]))

            X = X[:, channel_s1]
            X = (X - np.min(X, axis=2, keepdims=True)) / (
                    np.max(X, axis=2, keepdims=True) - np.min(X, axis=2, keepdims=True))
            Y = data['test_labels'].reshape(-1)
            return X, Y.astype(np.int64)

        X = data['train'].reshape((-1, data['train'].shape[2], data['train'].shape[3]))
        Y = self.change_labels(data['train_labels']).reshape(-1)

        if extend:
            positive_sample = X[Y == 1]
            for i in range(1):
                X = np.vstack((X, positive_sample))
                Y = np.concatenate((Y, np.ones(len(positive_sample), dtype=np.int)))

        X = X[:, channel_s1]
        X = (X - np.min(X, axis=2, keepdims=True)) / (
                np.max(X, axis=2, keepdims=True) - np.min(X, axis=2, keepdims=True))

        # tX = X[:60]
        # tY = Y[:60]
        # X = X[60:]
        # Y = Y[60:]
        shuff_index = np.array(range(len(X)))
        shuff_index = np.random.permutation(shuff_index)
        X = X[shuff_index]
        Y = Y[shuff_index]

        train_len = int(len(Y) * (9/10))
        if full:
            return X, Y.astype(np.int64)
        if train:
            # return X, Y.astype(np.int64)
            X = X[:train_len]
            Y = Y[:train_len]
        else:
            # return tX, tY.astype(np.int64)
            X = X[train_len:]
            Y = Y[train_len:]

        return X, Y.astype(np.int64)


class SleepSignal(Dataset):

    def __init__(self, root, datas, labels, train=True, transform=None, full=None, extend=None, test=False):
        super(SleepSignal, self).__init__()

        self.train = train
        self.transform = transform

        self.data, self.label = self.load_ss(datas, labels, self.train)

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        if self.transform is not None:
            data = self.transform(data)

        return data.type(torch.float32), label - 2

    def __len__(self):

        return len(self.label)

    def load_ss(self, datas, labels, train=True):

        X = datas
        Y = labels
        train_len = 2500
        if train:
            X = X[:train_len]
            Y = Y[:train_len]
        else:
            X = X[train_len:]
            Y = Y[train_len:]

        return X, Y.astype(np.int64)


DATASET_FN_DICT = {
    'p300': P300,
    'ss': SleepSignal,
}


dataset_list = DATASET_FN_DICT.keys()


def _get_dataset(dataset_name='p300'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


# get the loader of all datas
def get_dataloader(dataset_path='S1',
                   dataset_name='p300', train=True, batch_size=50, shuffle=True, full=False, extend=False, datas=None, labels=None, test=False):
    dataset_path = '{}/{}'.format(DATASETS_DIR, dataset_path)
    dataset = _get_dataset(dataset_name)

    transform = [
        lambda x: torch.tensor(x),
        # lambda x: x / torch.norm(x, dim=1, keepdim=True),
        # lambda x: (x - torch.min(x, dim=1, keepdim=True)) / (torch.max(x, dim=1, keepdim=True) - torch.min(x, dim=1, keepdim=True))
    ]
    loader = DataLoader(
        dataset(dataset_path, train=train, transform=transforms.Compose(transform), full=full, extend=extend, datas=datas, labels=labels, test=test),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader
