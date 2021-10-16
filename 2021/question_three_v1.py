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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():

    INPUT_DIM = 21
    OUTPUT_DIM = 6
    BATCH_SIZE = 64
    EPOCH = 500
    data_path = "C"

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
        for index, (predict_feature, label) in enumerate(dataloader):

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
        model.eval()
        with torch.no_grad():
            valid_predict, valid_label = next(iter(valid_loader))
            valid_predict, valid_label = valid_predict.to(DEVICE), valid_label.to(DEVICE)
            t_logit = model(valid_predict)
            t_loss = criterion(t_logit, valid_label)
            print("epoch: {}, training_loss: {}; validate_loss: {}".format(i+1, total_loss / len(dataloader), t_loss))
            loss_arr_valid.append(t_loss)

    plt.figure(figsize=(6.4, 5.2))
    plt.rcParams["font.family"] = 'Arial Unicode MS'
    plt.plot(np.array(loss_arr), 'orange', label='training loss', linewidth=3)
    plt.plot(np.array(loss_arr_valid), 'royalblue', label='validate loss', linewidth=3)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig('./lossC.png', dpi=600)
    torch.save(model, './RegressionC.pk')


if __name__ == '__main__':

    import time
    begin = time.time()
    main()
    end = time.time()

    print(end - begin)
