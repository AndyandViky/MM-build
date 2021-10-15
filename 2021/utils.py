# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: utils.py
@time: 2021/10/14 12:18
@desc: utils.py
'''
import numpy as np


def max_min_normalization(data: np.ndarray, axis: int = 0):
    mins = np.min(data, axis=axis, keepdims=True)
    maxs = np.max(data, axis=axis, keepdims=True)
    nor_data = (data - mins) / (maxs - mins + 1e-10)
    return nor_data


def L2_normalization(data: np.ndarray, axis: int = 0):
    nor_data = data / (np.linalg.norm(data, axis=axis, keepdims=True) + 1e-10)
    return nor_data


def creat_dataset(norm_data, data, look_back):
    """
    create a dataset for LSTM
    :param dataset:
    :param look_back:
    :return:
    """
    data_x = []
    data_y = []
    for i in range(len(norm_data)-look_back):
        data_x.append(norm_data[i:i+look_back])
        data_y.append(data[i+look_back])
    return np.asarray(data_x), np.asarray(data_y)