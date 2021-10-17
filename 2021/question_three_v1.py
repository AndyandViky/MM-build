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
from models import Regression
from utils import calculate_aqi, get_predict_input
from sklearn.metrics import accuracy_score, f1_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():

    INPUT_DIM = 21
    OUTPUT_DIM = 6
    BATCH_SIZE = 64
    EPOCH = 500
    data_path = "A"


    criterion = nn.MSELoss().to(DEVICE)
    model = Regression(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    optim = Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.99))

    dataloader = get_dataloader(-1, dataset_path=data_path, train=True, batch_size=BATCH_SIZE,
                                dataset_name="AirQualityV1")
    valid_loader = get_dataloader(-1, dataset_path=data_path, train=False, batch_size=5000, shuffle=False,
                                  dataset_name="AirQualityV1")

    loss_arr = []
    loss_arr_valid = []
    for i in range(EPOCH):
        # training
        total_loss = 0
        for index, (predict_feature, label, _, _) in enumerate(dataloader):

            model.train()
            predict_feature, label = predict_feature.to(DEVICE), label.to(DEVICE)
            logit = model(predict_feature)

            loss = criterion(logit, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
        loss_arr.append(total_loss / len(dataloader))

        # valid
        # model.eval()
        # with torch.no_grad():
        #     valid_predict, valid_label, predict_feature, target_feature = next(iter(valid_loader))
        #     valid_predict, valid_label = valid_predict.to(DEVICE), valid_label.to(DEVICE)
        #     t_logit = model(valid_predict)
        #     t_loss = criterion(t_logit, valid_label)
        #     t_aqi, t_index = calculate_aqi(t_logit.data.cpu().numpy() + predict_feature.data.numpy())
        #     r_aqi, r_index = calculate_aqi(target_feature.data.numpy())
        #     acc = accuracy_score(r_index, t_index)
        #     f1 = f1_score(r_index, t_index, average="macro")
        #     print("epoch: {}, training_loss: {}; validate_loss: {}, acc: {:.4f}, f1: {:.4f}"
        #           .format(i+1, total_loss / len(dataloader), t_loss, acc, f1))
        #     loss_arr_valid.append(t_loss)
    #
    #         if (i + 1) % 50 == 0:
    #             plt.figure(figsize=(10, 10))
    #             y_labels = ["SO2(μg/m³)", "NO2(μg/m³)", "PM10(μg/m³)", "PM2.5(μg/m³)", "O3(μg/m³)", "CO(mg/m³)"]
    #             for item in range(1, 5):
    #                 plt.rcParams["font.family"] = 'Arial Unicode MS'
    #                 plt.subplot(2, 2, item)
    #                 plt.plot(np.array(r_aqi)[:, item], 'orange', label='real AQI', linewidth=3)
    #                 plt.plot(np.array(t_aqi)[:, item], 'royalblue', label='predict AQI', linewidth=3)
    #                 plt.ylabel("AQI ({})".format(y_labels[item]))
    #                 plt.xlabel("Time")
    #                 plt.legend()
    #                 plt.xticks(rotation=45)
    #             plt.savefig('./results/AQI{}_v1.png'.format(data_path), dpi=200)
    #
    # plt.figure(figsize=(6.4, 5.2))
    # plt.rcParams["font.family"] = 'Arial Unicode MS'
    # plt.plot(np.array(loss_arr), 'orange', label='training loss', linewidth=3)
    # plt.plot(np.array(loss_arr_valid), 'royalblue', label='validate loss', linewidth=3)
    # plt.ylabel('MSE')
    # plt.xlabel('Epoch')
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.savefig('./results/loss{}_v1.png'.format(data_path), dpi=200)
    # torch.save(model, './results/Regression{}.pk'.format(data_path))

        if (i + 1) % 50 == 0:
            test(model, data_path)


def test(model: Regression, data_path: str) -> None:
    predict_feature_norm, predict_feature = get_predict_input(data_path)
    predict_feature_norm = torch.tensor(predict_feature_norm).type(torch.float32).to(DEVICE)
    logit = model(predict_feature_norm)
    output = logit.data.cpu().numpy() + predict_feature
    aqi, index = calculate_aqi(output, True)
    print("aqi: {}, index: {}".format(np.max(aqi, axis=1), index))


if __name__ == '__main__':

    import time
    begin = time.time()
    main()
    end = time.time()
    print(end - begin)
