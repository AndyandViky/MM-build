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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():

    PRE_INPUT_DIM = 21
    TAR_INPUT_DIM = 6
    LOOK_BACK = 4
    OUTPUT_DIM = 6
    BATCH_SIZE = 64
    HIDDEN_DIM = 64
    EPOCH = 1000
    data_path = "C"

    criterion = nn.MSELoss(reduction='mean').to(DEVICE)
    model = SequentialRegression(PRE_INPUT_DIM, TAR_INPUT_DIM, LOOK_BACK, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    optim = Adam(model.parameters(), lr=1e-2, betas=(0.5, 0.99))

    dataloader = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path, train=True, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader(look_back=LOOK_BACK, dataset_path=data_path, train=False, batch_size=5000, shuffle=False)

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

        # valid
        model.eval()
        with torch.no_grad():
            valid_predict, valid_target, valid_label = next(iter(valid_loader))
            valid_predict, valid_target, valid_label = valid_predict.to(DEVICE), valid_target.to(DEVICE), valid_label.to(DEVICE)
            t_logit = model(valid_predict, valid_target)
            t_loss = criterion(t_logit, valid_label)
            print("epoch: {}, training_loss: {}; validate_loss: {}".format(i+1, total_loss / len(dataloader), t_loss))
        if (i + 1) % 50 == 0:
            plt.figure(figsize=(12, 7.5))
            for item in range(6):
                plt.rcParams["font.family"] = 'Arial Unicode MS'
                plt.subplot(3, 2, item + 1)
                plt.plot(valid_label.data.cpu().numpy()[:, item], 'b')
                plt.plot(t_logit.data.cpu().numpy()[:, item], 'r')
                plt.ylabel(u'训练过程')
                plt.xticks(rotation=45)
            plt.savefig('./result.png', dpi=600)

    torch.save(model, './SequentialRegression.pk')


def test():
    INPUT_DIM = 150
    OUTPUT_DIM = 2
    HIDDEN_DIM = 32

    model = SequentialRegression(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load('./SequentialRegression.pk'))
    test_loader = get_dataloader(dataset_path='A', test=True, batch_size=5000, shuffle=False)

    # testing
    model.eval()
    with torch.no_grad():
        t_data, t_label = next(iter(test_loader))
        t_data = t_data.to(DEVICE)


if __name__ == '__main__':

    import time
    begin = time.time()
    main()
    end = time.time()

    print(end - begin)
    # test()

