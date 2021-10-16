# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: datasets.py
@time: 2021/10/14 12:22
@desc: datasets.py
'''
import os
import pandas as pd
import numpy as np
import scipy.io as scio
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from config import DATASETS_DIR
from utils import L2_normalization, creat_dataset, max_min_normalization


def preprocess(root: str):
    reader = pd.ExcelFile(root)
    sheet_names = reader.sheet_names
    predict_df = pd.read_excel(root, sheet_name=sheet_names[0])
    predict_df.drop(predict_df.columns[0], axis=1, inplace=True)
    predict_df.drop(predict_df.columns[1], axis=1, inplace=True)
    predict_df = predict_df.drop_duplicates('预测时间', keep='first')

    target_df = pd.read_excel(root, sheet_name=sheet_names[1])
    target_df.drop(target_df.columns[1], axis=1, inplace=True)
    # target_df = target_df.replace("—", 0.0)
    for index, row in target_df.iteritems():
        if index == '监测时间':
            continue
        target_df[index] = target_df[index].replace("—", target_df[index].mean())
        target_df[index].fillna(target_df[index].mean(), inplace=True)
    # target_df.drop(index=list(range(11036)), inplace=True)

    total_df = pd.merge(predict_df, target_df, how="inner", left_on=predict_df['预测时间'], right_on=target_df['监测时间'])
    total_df.drop(total_df.columns[0], axis=1, inplace=True)
    total_df.to_csv("./datasets/A3.csv", index=None)
# preprocess('{}/{}.xlsx'.format(DATASETS_DIR, 'A3'))


class AirQuality(Dataset):

    def __init__(self, root, look_back: int, train=True, transform=None):
        super(AirQuality, self).__init__()

        self.root = root
        self.train = train
        self.look_back = look_back
        self.transform = transform

        self.pre_data, self.tar_data, self.label = self._get_data_from_csv(root, look_back, train)

    def __getitem__(self, index):
        pre, tar, label = self.pre_data[index], self.tar_data[index], self.label[index]
        if self.transform is not None:
            pre = self.transform(pre)
            tar = self.transform(tar)
            label = self.transform(label)

        return pre.type(torch.float32), tar.type(torch.float32), label.type(torch.float32)

    def __len__(self):
        return len(self.label)

    def _get_data_from_csv(self, root: str, look_back: int, train: bool = True):
        training_rate = 0.8
        data = pd.read_csv(root).values
        predict_features = data[:, 1:22].astype(np.float)
        target_features = data[:, 23:29].astype(np.float)
        predict_features_norm = L2_normalization(predict_features, axis=1)
        target_features_norm = L2_normalization(target_features, axis=1)
        predict_features, predict_label = creat_dataset(predict_features_norm, predict_features, look_back)
        target_features, target_label = creat_dataset(target_features_norm, target_features, look_back)
        # target_label /= np.array([1, 5, 5, 3, 10, 1, 5, 10, 100, 1, 10])
        # 用 predict 的特征，结合真实 label 进行预测，这样在预测 13 号数据时输入 13 号的预测特征进行辅助。
        p_l, t_l = len(predict_features), len(target_features)
        if train:
            return predict_features[: int(p_l * training_rate)], \
                   target_features[: int(p_l * training_rate)], \
                   target_label[: int(p_l * training_rate)]
        else:
            return predict_features[int(p_l * training_rate) + look_back:], \
                   target_features[int(p_l * training_rate) + look_back:], \
                   target_label[int(p_l * training_rate) + look_back:]


class AirQualityV1(Dataset):

    def __init__(self, root, train=True, transform=None, look_back=None):
        super(AirQualityV1, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform

        self.pre_data, self.tar_label = self._get_data_from_csv(root, train)

    def __getitem__(self, index):
        pre, label = self.pre_data[index], self.tar_label[index]
        if self.transform is not None:
            pre = self.transform(pre)
            label = self.transform(label)
        return pre.type(torch.float32), label.type(torch.float32)

    def __len__(self):
        return len(self.tar_label)

    def _get_data_from_csv(self, root: str, train: bool = True):
        training_rate = 0.5
        data = pd.read_csv(root).values
        predict_features = data[:, 1:22].astype(np.float)
        target_features = data[:, 23:29].astype(np.float)
        target_label = target_features - predict_features[:, 15:]
        predict_features = L2_normalization(predict_features, axis=1)
        # 用 predict 的特征，结合真实 label 进行预测，这样在预测 13 号数据时输入 13 号的预测特征进行辅助。
        p_l, t_l = len(predict_features), len(target_features)
        if train:
            return predict_features[: int(p_l * training_rate)], \
                   target_label[: int(p_l * training_rate)]
        else:
            return predict_features[int(p_l * training_rate):], \
                   target_label[int(p_l * training_rate):]


DATASET_FN_DICT = {
    'AirQuality': AirQuality,
    'AirQualityV1': AirQualityV1,
}


dataset_list = DATASET_FN_DICT.keys()


def _get_dataset(dataset_name='AirQuality'):
    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


# get the loader of all datas
def get_dataloader(look_back: int, dataset_path='A',
                   dataset_name='AirQuality', train=True, batch_size=50, shuffle=True):
    dataset_path = '{}/{}.csv'.format(DATASETS_DIR, dataset_path)
    dataset = _get_dataset(dataset_name)

    transform = [
        lambda x: torch.tensor(x),
    ]
    loader = DataLoader(
        dataset(dataset_path, look_back=look_back, train=train, transform=transforms.Compose(transform)),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader

# get_dataloader()