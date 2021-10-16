# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: question_four.py
@time: 2021/10/14 12:18
@desc: question_four.py
'''
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from datasets import get_dataloader
from config import DEVICE
from models import SequentialRegressionMutiArea
from itertools import cycle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():

    PRE_INPUT_DIM = 21
    TAR_INPUT_DIM = 6
    LOOK_BACK = 4
    OUTPUT_DIM = 6
    BATCH_SIZE = 64
    HIDDEN_DIM = 64
    EPOCH = 1000
    data_path = ['A', 'A1', 'A2', 'A3']

    criterion = nn.MSELoss(reduction='mean').to(DEVICE)
    model = SequentialRegressionMutiArea(PRE_INPUT_DIM, TAR_INPUT_DIM, LOOK_BACK, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    optim = Adam(model.parameters(), lr=1e-2, betas=(0.5, 0.99))

    dataloaderA = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[0], train=True, batch_size=BATCH_SIZE)
    valid_loaderA = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[0], train=False, batch_size=5000,
                                  shuffle=False)

    dataloaderA1 = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[1], train=True, batch_size=BATCH_SIZE)
    valid_loaderA1 = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[1], train=False, batch_size=5000,
                                  shuffle=False)

    dataloaderA2 = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[2], train=True, batch_size=BATCH_SIZE)
    valid_loaderA2 = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[2], train=False, batch_size=5000,
                                  shuffle=False)

    dataloaderA3 = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[3], train=True, batch_size=BATCH_SIZE)
    valid_loaderA3 = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path[3], train=False, batch_size=5000,
                                  shuffle=False)

    for i in range(EPOCH):
        # training
        total_loss = 0
        for index, data in enumerate(zip(cycle(dataloaderA), dataloaderA1, cycle(dataloaderA2), cycle(dataloaderA3))):
            model.train()
            preA, tarA, labelA = data[0]
            preA1, tarA1, labelA1 = data[1]
            preA2, tarA2, labelA2 = data[2]
            preA3, tarA3, labelA3 = data[3]
            preA, tarA, labelA = preA.to(DEVICE), tarA.to(DEVICE), labelA.to(DEVICE)
            preA1, tarA1, labelA1 = preA1.to(DEVICE), tarA1.to(DEVICE), labelA1.to(DEVICE)
            preA2, tarA2, labelA2 = preA2.to(DEVICE), tarA2.to(DEVICE), labelA2.to(DEVICE)
            preA3, tarA3, labelA3 = preA3.to(DEVICE), tarA3.to(DEVICE), labelA3.to(DEVICE)

            logitA, logitA1, logitA2, logitA3 = model((preA, tarA), (preA1, tarA1), (preA2, tarA2), (preA3, tarA3))

            # 多 loss，多任务监督
            # 分通用数据层 和 任务相关层
            loss1 = criterion(logitA, labelA)
            loss2 = criterion(logitA1, labelA1)
            loss3 = criterion(logitA2, labelA2)
            loss4 = criterion(logitA3, labelA3)
            loss = loss1 + loss2 + loss3 + loss4

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        # valid
        model.eval()
        with torch.no_grad():
            preA, tarA, labelA = next(iter(valid_loaderA))
            preA1, tarA1, labelA1 = next(iter(valid_loaderA1))
            preA2, tarA2, labelA2 = next(iter(valid_loaderA2))
            preA3, tarA3, labelA3 = next(iter(valid_loaderA3))

            preA, tarA, labelA = preA.to(DEVICE), tarA.to(DEVICE), labelA.to(DEVICE)
            preA1, tarA1, labelA1 = preA1.to(DEVICE), tarA1.to(DEVICE), labelA1.to(DEVICE)
            preA2, tarA2, labelA2 = preA2.to(DEVICE), tarA2.to(DEVICE), labelA2.to(DEVICE)
            preA3, tarA3, labelA3 = preA3.to(DEVICE), tarA3.to(DEVICE), labelA3.to(DEVICE)

            logitA, logitA1, logitA2, logitA3 = model((preA, tarA), (preA1, tarA1), (preA2, tarA2), (preA3, tarA3))

            # 多 loss，多任务监督
            # 分通用数据层 和 任务相关层
            loss1 = criterion(logitA, labelA)
            loss2 = criterion(logitA1, labelA1)
            loss3 = criterion(logitA2, labelA2)
            loss4 = criterion(logitA3, labelA3)
            loss = loss1 + loss2 + loss3 + loss4

            print("epoch: {}, training_loss: {}; validate_loss: {}".format(i+1, total_loss / len(dataloaderA), loss))
        if (i + 1) % 50 == 0:
            plt.figure(figsize=(12, 7.5))
            for item in range(6):
                plt.rcParams["font.family"] = 'Arial Unicode MS'
                plt.subplot(3, 2, item + 1)
                plt.plot(labelA1.data.cpu().numpy()[:, item], 'b')
                plt.plot(logitA1.data.cpu().numpy()[:, item], 'r')
                plt.ylabel(u'训练过程')
                plt.xticks(rotation=45)
            plt.savefig('./result.png', dpi=600)

    torch.save(model, './SequentialRegressionMutiArea.pk')


if __name__ == '__main__':

    import time
    begin = time.time()
    main()
    end = time.time()
    print(end - begin)
