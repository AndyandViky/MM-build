# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: question_two.py
@time: 2020/9/19 19:59
@desc: question_two.py
'''
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Tuple
from torch.optim import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from datasets import get_dataloader
from config import DEVICE


class ChannelFilter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2):
        super(ChannelFilter, self).__init__()

        self.filter = nn.Conv1d(input_dim, 64, 20, 2)
        self.cnn = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, 3, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 64, 3, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, 3, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
        )
        self.out = nn.Linear(2, output_dim)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.dropout = nn.Dropout(0.1)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x: Tensor) -> Tensor:

        output = self.attention(x)
        output = self.out(output)
        return output

    def attention(self, x: Tensor) -> Tensor:

        x = self.filter(x)
        output = self.cnn(x)
        output = self.dropout(output)
        u = torch.tanh(torch.matmul(output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = torch.softmax(att, dim=1)
        scored_x = output * att_score
        output = torch.sum(scored_x, dim=1)
        return output


def main():

    INPUT_DIM = 20
    OUTPUT_DIM = 2
    BATCH_SIZE = 10
    EPOCH = 60

    criterion = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    channelFilter = ChannelFilter(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    optim = Adam(channelFilter.parameters(), lr=1e-3, betas=(0.5, 0.99))

    dataloader = get_dataloader(dataset_path='S1', train=True, batch_size=BATCH_SIZE, full=True)
    valid_loader = get_dataloader(dataset_path='S1', train=False, batch_size=5000, shuffle=False)

    for i in range(EPOCH):
        # training
        total_loss = 0
        for index, (data, label) in enumerate(dataloader):

            channelFilter.train()
            data, label = data.to(DEVICE), label.to(DEVICE)
            logit = channelFilter(data)

            loss = criterion(logit, label) / BATCH_SIZE
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        # valid
        channelFilter.eval()
        with torch.no_grad():
            valid_data, valid_labels = valid_loader.dataset.data, valid_loader.dataset.label
            valid_data = torch.tensor(valid_data, dtype=torch.float32).to(DEVICE)
            valid_labels = torch.tensor(valid_labels).to(DEVICE)
            t_logit = channelFilter(valid_data)
            t_loss = criterion(t_logit, valid_labels)
            valid_pred = torch.argmax(t_logit, dim=1).data.cpu().numpy()

            valid_labels = valid_labels.data.cpu().numpy()
            acc, p, r, f1 = accuracy_score(valid_labels, valid_pred), precision_score(valid_labels, valid_pred), \
                            recall_score(valid_labels, valid_pred), f1_score(valid_labels, valid_pred)
            print('iter: {}, train_loss: {:.3f}, valid_loss: {:.3f}, acc: {:.3f}, p: {:.3f}, r: {:.3f}, f1: {:.3f}'.format(i, total_loss / len(dataloader), t_loss,
                                                                                             acc, p, r, f1))
    torch.save(channelFilter.state_dict(), './channelFilter.pk')


if __name__ == '__main__':

    # test
    # channelFilter = ChannelFilter(20, 2).to(DEVICE)
    # channelFilter.load_state_dict(torch.load('./channelFilter.pk'))
    #
    # filter = channelFilter.filter.weight

    import time
    begin = time.time()
    main()
    end = time.time()

    print(end - begin)
