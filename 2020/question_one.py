# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: question_one.py
@time: 2020/9/17 13:50
@desc: question_one.py
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


class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Classifier, self).__init__()

        self.lstm = nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, output_dim)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        self.dropout = nn.Dropout(0.1)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x: Tensor) -> Tensor:

        output = self.attention(x)
        output = self.out(output)
        return output

    def attention(self, x: Tensor) -> Tensor:

        output, hidden = self.lstm(x)
        output = self.dropout(output)
        u = torch.tanh(torch.matmul(output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = torch.softmax(att, dim=1)
        scored_x = output * att_score
        output = torch.sum(scored_x, dim=1)
        return output


def main():

    INPUT_DIM = 150
    OUTPUT_DIM = 2
    BATCH_SIZE = 10
    HIDDEN_DIM = 32
    EPOCH = 50

    criterion = nn.CrossEntropyLoss(reduction='sum').to(DEVICE)
    classifier = Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    optim = Adam(classifier.parameters(), lr=1e-3, betas=(0.5, 0.99))

    dataloader = get_dataloader(dataset_path='S1', train=True, batch_size=BATCH_SIZE, full=False)
    valid_loader = get_dataloader(dataset_path='S1', train=False, batch_size=5000, shuffle=False)

    best_score = 0
    best_model = None
    for i in range(EPOCH):
        # training
        total_loss = 0
        for index, (data, label) in enumerate(dataloader):

            classifier.train()
            data, label = data.to(DEVICE), label.to(DEVICE)
            logit = classifier(data)

            loss = criterion(logit, label) / BATCH_SIZE
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        # valid
        classifier.eval()
        with torch.no_grad():
            valid_data, valid_labels = valid_loader.dataset.data, valid_loader.dataset.label
            valid_data = torch.tensor(valid_data, dtype=torch.float32).to(DEVICE)
            valid_labels = torch.tensor(valid_labels).to(DEVICE)
            t_logit = classifier(valid_data)
            t_loss = criterion(t_logit, valid_labels)
            valid_pred = torch.argmax(t_logit, dim=1).data.cpu().numpy()

            valid_labels = valid_labels.data.cpu().numpy()
            acc, p, r, f1 = accuracy_score(valid_labels, valid_pred), precision_score(valid_labels, valid_pred), \
                            recall_score(valid_labels, valid_pred), f1_score(valid_labels, valid_pred)
            print('iter: {}, train_loss: {:.3f}, valid_loss: {:.3f}, acc: {:.3f}, p: {:.3f}, r: {:.3f}, f1: {:.3f}'.format(i, total_loss / len(dataloader), t_loss,
                                                                                            acc, p, r, f1))
            if f1 > best_score:
                best_score = f1
                best_model = classifier.state_dict()
    torch.save(best_model, './classifier.pk')


def test():
    INPUT_DIM = 150
    OUTPUT_DIM = 2
    HIDDEN_DIM = 32

    classifier = Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    classifier.load_state_dict(torch.load('./classifier.pk'))
    test_loader = get_dataloader(dataset_path='S1', test=True, batch_size=5000, shuffle=False)

    # testing
    classifier.eval()
    with torch.no_grad():
        t_data, t_label = next(iter(test_loader))
        t_data = t_data.to(DEVICE)
        t_logit = torch.softmax(classifier(t_data), dim=1)
        t_pred = torch.argmax(t_logit, dim=1).data.cpu().numpy()
        for i in range(10):
            s = t_pred[i*60: (i+1) * 60].reshape((5, 12))
            t = t_label[i*60: (i+1) * 60].reshape((5, 12)).data.cpu().numpy()
            t1 = t[s == 1]

            row = t1[t1 <= 6]
            if len(row) == 0:
                r_row = 0
            else:
                r_row = np.argmax(np.bincount(row))
            col = t1[t1 > 6]
            if len(col) == 0:
                r_col = 0
            else:
                r_col = np.argmax(np.bincount(col))
            print(r_row, r_col)


if __name__ == '__main__':

    import time
    begin = time.time()
    main()
    end = time.time()

    print(end - begin)
    test()
