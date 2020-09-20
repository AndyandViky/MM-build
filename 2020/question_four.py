# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: question_four.py
@time: 2020/9/18 19:55
@desc: question_four.py
'''
import pandas as pd
import numpy as np

from config import DATASETS_DIR
from xgb import selfTraining
from dnn import main as DNN_Training


def preprocess():

    names = ['清醒期（6）', '快速眼动期（5）', '睡眠I期（4）', '睡眠II期（3）', '深睡眠期（2）']
    datas = []
    labels = []
    for index, item in enumerate(names):
        data = pd.read_excel('{}/four.xlsx'.format(DATASETS_DIR), sheet_name=item).values
        label = data[:, 0]
        if index == 0:
            data = data[:, 1:5].astype(np.float32)
        else:
            data = data[:, 1:]
        datas.append(data)
        labels.append(label)

    datas = np.vstack(datas)
    labels = np.concatenate(labels, 0).astype(np.int)
    np.random.seed(172)
    shuff_index = np.array(range(len(datas)))
    shuff_index = np.random.permutation(shuff_index)
    datas = datas[shuff_index]
    labels = labels[shuff_index]
    return datas, labels


def main():

    len = 2500
    datas, labels = preprocess()

    # dnn
    DNN_Training(datas, labels, 100)

    # xgboost
    selfTraining(datas[:len], datas[len:], labels[:len], labels[len:])


if __name__ == '__main__':
    main()