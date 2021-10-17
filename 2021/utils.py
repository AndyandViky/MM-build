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
import pandas as pd
import numpy as np

from typing import Tuple


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


def calculate_aqi(data: np.ndarray, verbose: bool = False) -> Tuple:
    # 所有的AOI
    data = pd.DataFrame(data)
    res = []
    for i in range(int(len(data) / 24)):
        o3 = data.iloc[i * 24:(i + 1) * 24, 4].values
        polution = data.iloc[i * 24:(i + 1) * 24, [0, 1, 2, 3, 5]]

        maxO3 = o3_max_slide_ave(o3)
        avePol = polution_ave(polution)
        avePol.insert(4, maxO3)
        res.append(avePol)
    if verbose:
        print("平均浓度: {}".format(res))

    table = [[0, 50, 150, 475, 800, 1600, 2100, 2620],
             [0, 40, 80, 180, 280, 565, 750, 940],
             [0, 50, 150, 250, 350, 420, 500, 600],
             [0, 35, 75, 115, 150, 250, 350, 500],
             [0, 100, 160, 215, 265, 800],
             [0, 2, 4, 14, 24, 36, 48, 60],
             [0, 50, 100, 150, 200, 300, 400, 500]]
    ans = []
    index = []
    for i in range(int(len(data) / 24)):
        ele = []
        for j in range(6):
            item = res[i][j]
            itemList = table[j]
            for k in range(len(itemList)):
                if itemList[k] <= item < itemList[k + 1]:
                    BP_hi = itemList[k + 1]
                    BP_lo = itemList[k]
                    IAQI_hi = table[6][k + 1]
                    IAQI_lo = table[6][k]
                    IAQI_p = (IAQI_hi - IAQI_lo) / (BP_hi - BP_lo) * (item - BP_lo) + IAQI_lo
                    ele.append(IAQI_p)
                    break

        ans.append(ele)
        index.append(np.argmax(np.array(ele)))
    return ans, index


# o3 8小时滑动平均
def o3_max_slide_ave(o3):
    max = -100
    for i in range(8, 25):
        if i == 24:
            # temp = np.sum(o3[i-7: i])
            # temp = temp + o3[0]
            sum = np.sum(o3[i-7: i]) + o3[0]
        else:
            sum = np.sum(o3[i-7: i+1])
        ave = sum / 8
        if ave > max:
            max = ave
    return max


# 5个污染物24小时平均
def polution_ave(polution):
    ans = []
    for i in range(5):
        item = np.average(polution.iloc[:, i])
        if item >= 1:
            item = round(item)
        else:
            item = round(item, 1)
        ans.append(item)
    return ans


def get_predict_input(file_name: str, times: int=72) -> Tuple:
    predict_df = pd.read_excel("./datasets/{}.xlsx".format(file_name))
    predict_df.drop(predict_df.columns[0], axis=1, inplace=True)
    predict_df.drop(predict_df.columns[1], axis=1, inplace=True)
    predict_df = predict_df.drop_duplicates('预测时间', keep='first')
    predict_feature = predict_df.values[-times:, 1:].astype(np.float)
    predict_feature_norm = L2_normalization(predict_feature, axis=1)
    return predict_feature_norm, predict_feature[:, 15:]
