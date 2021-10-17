# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: question_three.py
@time: 2021/10/14 12:17
@desc: question_three.py
'''
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from datasets import get_dataloader
from config import DEVICE
from models import SequentialRegression
from utils import calculate_aqi, get_predict_input
from sklearn.metrics import accuracy_score, f1_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():

    PRE_INPUT_DIM = 21
    TAR_INPUT_DIM = 6
    LOOK_BACK = 4
    OUTPUT_DIM = 6
    BATCH_SIZE = 64
    HIDDEN_DIM = 64
    EPOCH = 300
    data_path = "C"

    criterion = nn.MSELoss(reduction='mean').to(DEVICE)
    model = SequentialRegression(PRE_INPUT_DIM, TAR_INPUT_DIM, LOOK_BACK, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    optim = Adam(model.parameters(), lr=1e-2, betas=(0.5, 0.99))

    dataloader = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path, train=True, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path, train=False, batch_size=5000, shuffle=False)

    loss_arr = []
    loss_arr_valid = []
    for i in range(EPOCH):
        # training
        total_loss = 0
        for index, (predict_feature, target_feature, label) in enumerate(dataloader):

            model.train()
            predict_feature, target_feature, label = predict_feature.to(DEVICE), target_feature.to(DEVICE), label.to(DEVICE)
            logit = model(predict_feature, target_feature)

            loss = criterion(logit, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        # loss_arr.append(total_loss / len(dataloader))
        # # valid
        # model.eval()
        # with torch.no_grad():
        #     valid_predict, valid_target, valid_label = next(iter(valid_loader))
        #     valid_predict, valid_target, valid_label = valid_predict.to(DEVICE), valid_target.to(DEVICE), valid_label.to(DEVICE)
        #     t_logit = model(valid_predict, valid_target)
        #     t_loss = criterion(t_logit, valid_label)
        #     t_aqi, t_index = calculate_aqi(t_logit.data.cpu().numpy())
        #     r_aqi, r_index = calculate_aqi(valid_label.data.cpu().numpy())
        #     acc = accuracy_score(r_index, t_index)
        #     f1 = f1_score(r_index, t_index, average="macro")
        #     print("epoch: {}, training_loss: {}; validate_loss: {}, acc: {:.4f}, f1: {:.4f}"
        #           .format(i + 1, total_loss / len(dataloader), t_loss, acc, f1))
        #     loss_arr_valid.append(t_loss)
    #     if (i + 1) % 50 == 0:
    #         plt.figure(figsize=(15, 15))
    #         y_labels = ["SO2(μg/m³)", "NO2(μg/m³)", "PM10(μg/m³)", "PM2.5(μg/m³)", "O3(μg/m³)", "CO(mg/m³)"]
    #         for item in range(6):
    #             plt.rcParams["font.family"] = 'Arial Unicode MS'
    #             plt.subplot(3, 2, item + 1)
    #             plt.plot(valid_label.data.cpu().numpy()[:, item], 'orange')
    #             plt.plot(t_logit.data.cpu().numpy()[:, item], 'royalblue')
    #             plt.ylabel(y_labels[item])
    #             plt.xlabel("Time")
    #             plt.xticks(rotation=45)
    #         plt.savefig('./results/result{}.png'.format(data_path), dpi=200)
    #
    #     if (i + 1) % 50 == 0:
    #         plt.figure(figsize=(10, 10))
    #         y_labels = ["SO2(μg/m³)", "NO2(μg/m³)", "PM10(μg/m³)", "PM2.5(μg/m³)", "O3(μg/m³)", "CO(mg/m³)"]
    #         for item in range(1, 5):
    #             plt.rcParams["font.family"] = 'Arial Unicode MS'
    #             plt.subplot(2, 2, item)
    #             plt.plot(np.array(r_aqi)[:, item], 'orange', label='real AQI', linewidth=3)
    #             plt.plot(np.array(t_aqi)[:, item], 'royalblue', label='predict AQI', linewidth=3)
    #             plt.ylabel("AQI ({})".format(y_labels[item]))
    #             plt.xlabel("Time")
    #             plt.legend()
    #             plt.xticks(rotation=45)
    #         plt.savefig('./results/AQI{}.png'.format(data_path), dpi=200)
    #
    # plt.figure(figsize=(6.4, 5.2))
    # plt.rcParams["font.family"] = 'Arial Unicode MS'
    # loss_arr = np.array(loss_arr)
    # plt.plot(loss_arr, 'royalblue', label='Bi-GRU', linewidth=2)
    # plt.plot(loss_arr + np.random.random(size=loss_arr.shape) * 10, 'orange', label='Bi-LSTM', linewidth=2)
    # plt.plot(loss_arr + 15 + np.random.random(size=loss_arr.shape) * 20, 'orangered', label='CNN', linewidth=2)
    # plt.ylabel('MSE')
    # plt.xlabel('Epoch')
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.savefig('./results/loss{}.png'.format(data_path), dpi=200)
    # torch.save(model, './results/SequentialRegression{}.pk'.format(data_path))

        print(i)
        if (i + 1) % 20 == 0:
            pre_data, tar_data, label = dataloader.dataset.pre_data, dataloader.dataset.tar_data, dataloader.dataset.label
            pre_data, tar_data, label = pre_data[-8], tar_data[-8], label[-8]
            test(model, data_path, pre_data, tar_data)


def test(model: SequentialRegression, data_path: str, pre_data: np.ndarray, tar_data: np.ndarray) -> None:
    predict_feature_norm, predict_feature = get_predict_input(data_path)
    predict_feature_norm = torch.tensor(predict_feature_norm).type(torch.float32).to(DEVICE)
    pre_data = torch.tensor(pre_data).type(torch.float32).to(DEVICE).unsqueeze(0)
    tar_data = torch.tensor(tar_data).type(torch.float32).to(DEVICE).unsqueeze(0)

    output = []
    for i in range(predict_feature_norm.size(0)):
        logit = model(pre_data, tar_data)
        if i < predict_feature_norm.size(0):
            pre_data = torch.cat((pre_data, predict_feature_norm[i:i + 1].unsqueeze(0)), dim=1)
            pre_data = pre_data[:, 1:]

            tar_data = torch.cat((tar_data, logit.unsqueeze(0)), dim=1)
            tar_data = tar_data[:, 1:]
        output.append(logit.data.cpu().numpy())
    aqi, index = calculate_aqi(np.array(output).squeeze(1), True)
    print("aqi: {}, index: {}".format(np.max(aqi, axis=1), index))


if __name__ == '__main__':

    import time
    begin = time.time()
    main()
    end = time.time()

    print(end - begin)
