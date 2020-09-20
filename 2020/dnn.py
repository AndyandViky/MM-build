# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: dnn.py
@time: 2020/9/19 8:20
@desc: dnn.py
'''
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Tuple
from torch.optim import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from datasets import get_dataloader
from config import DEVICE


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(True),
        )
        self.out = nn.Linear(64 * 2, 5)
        self.w_omega = nn.Parameter(torch.Tensor(64 * 2, 64 * 2))
        self.u_omega = nn.Parameter(torch.Tensor(64 * 2, 1))
        self.dropout = nn.Dropout(0.1)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x: Tensor) -> Tensor:

        output = self.model(x).view((x.size(0), 2, 128))
        output = self.attention(output)
        output = self.out(output)
        return output

    def attention(self, x: Tensor) -> Tensor:

        output = self.dropout(x)
        u = torch.tanh(torch.matmul(output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = torch.softmax(att, dim=1)
        scored_x = output * att_score
        output = torch.sum(scored_x, dim=1)
        return output


def main(datas, labels, epoch):

    classifier = DNN().to(DEVICE)
    dataloader = get_dataloader(dataset_name='ss', batch_size=50, train=True, datas=datas, labels=labels)
    test_dataloader = get_dataloader(dataset_name='ss', batch_size=5000, train=False, datas=datas, labels=labels)
    optim = Adam(classifier.parameters(), lr=1e-3, betas=(0.5, 0.99), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)

    best_score = 0
    best_model = None
    for i in range(epoch):
        total_loss = 0
        for index, (data, label) in enumerate(dataloader):
            classifier.train()
            data, label = data.to(DEVICE), label.to(DEVICE)
            logit = classifier(data)

            loss = criterion(logit, label) / 50
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        # testing
        classifier.eval()
        with torch.no_grad():
            t_data, t_label = next(iter(test_dataloader))
            t_data = t_data.to(DEVICE)
            t_label = t_label.to(DEVICE)
            t_logit = classifier(t_data)
            test_loss = criterion(t_logit, t_label) / t_data.size(0)
            t_pred = torch.argmax(t_logit, dim=1).data.cpu().numpy()
            t_label = t_label.data.cpu().numpy()
            acc, p, r, f1 = accuracy_score(t_label, t_pred), precision_score(t_label, t_pred, average='weighted'), \
                            recall_score(t_label, t_pred, average='weighted'), f1_score(t_label, t_pred,
                                                                                        average='weighted')
            if acc > best_score:
                best_score = acc
                best_model = classifier.state_dict()
            print(
                'iter: {}, test_loss: {:.3f}, acc: {:.3f}, p: {:.3f}, r: {:.3f}, f1: {:.3f}'.format(i, test_loss.item(), acc, p,
                                                                                          r, f1))

    from plot import Q4
    classifier.load_state_dict(best_model)
    classifier.eval()
    with torch.no_grad():
        t_data, t_label = next(iter(test_dataloader))
        t_data = t_data.to(DEVICE)
        t_label = t_label.to(DEVICE)
        t_logit = classifier(t_data)
        t_pred = torch.argmax(t_logit, dim=1).data.cpu().numpy()
        t_label = t_label.data.cpu().numpy()
        cm = confusion_matrix(t_label, t_pred)
        Q4().confusion_matrix(cm)