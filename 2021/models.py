# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: models.py
@time: 2021/10/14 12:18
@desc: models.py
'''
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Tuple

class SequentialRegression(nn.Module):
    def __init__(self, pre_input_dim: int, tar_input_dim: int, seq_len: int, hidden_dim: int, output_dim: int):
        super(SequentialRegression, self).__init__()

        self.predict_projection = nn.GRU(pre_input_dim, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.target_projection = nn.GRU(tar_input_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

        self.out = nn.Linear(hidden_dim * 2, output_dim)

        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        self.dropout = nn.Dropout(0.1)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, predict_feature: Tensor, target_feature: Tensor) -> Tensor:
        pre_out = self.predict_projection(predict_feature)[0]
        tar_out = self.target_projection(target_feature)[0]

        inner_att_pre = self.inner_attention(pre_out)
        inner_att_tar = self.inner_attention(tar_out)

        output = self.out(self.outer_attention(inner_att_pre, inner_att_tar))
        # output = self.out(inner_att_pre + inner_att_tar)
        return output

    def outer_attention(self, predict_feature: Tensor, target_feature: Tensor) -> Tensor:
        att_logit = torch.sum(torch.bmm(target_feature, predict_feature.permute((0, 2, 1))), dim=2)
        att_score = torch.softmax(att_logit, dim=1).unsqueeze(2)
        att_out = att_score * predict_feature
        target_feature = target_feature + att_out
        return torch.sum(target_feature, dim=1)

    def inner_attention(self, output: Tensor) -> Tensor:
        output = self.dropout(output)
        u = torch.tanh(torch.matmul(output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = torch.softmax(att, dim=1)
        output = output * att_score
        return output


class SequentialRegressionMutiArea(SequentialRegression):
    def __init__(self, pre_input_dim: int, tar_input_dim: int, seq_len: int, hidden_dim: int, output_dim: int):
        super(SequentialRegressionMutiArea, self).__init__(
            pre_input_dim,
            tar_input_dim,
            seq_len,
            hidden_dim,
            output_dim,
        )

        self.task1 = nn.Linear(hidden_dim * 2, output_dim)
        self.task2 = nn.Linear(hidden_dim * 2, output_dim)
        self.task3 = nn.Linear(hidden_dim * 2, output_dim)
        self.task4 = nn.Linear(hidden_dim * 2, output_dim)

    def get_area_embedding(self, area: Tuple) -> Tensor:
        pre_out = self.predict_projection(area[0])[0]
        tar_out = self.target_projection(area[1])[0]
        inner_att_pre = self.inner_attention(pre_out)
        inner_att_tar = self.inner_attention(tar_out)

        output = self.outer_attention(inner_att_pre, inner_att_tar)
        return output

    def forward(self, area1: Tuple, area2: Tuple, area3: Tuple, area4: Tuple) -> Tuple:
        out1 = self.get_area_embedding(area1)
        out2 = self.get_area_embedding(area2)
        out3 = self.get_area_embedding(area3)
        out4 = self.get_area_embedding(area4)

        # task_related_module
        logit1 = self.task1(out1)
        logit2 = self.task1(out2)
        logit3 = self.task1(out3)
        logit4 = self.task1(out4)

        return logit1, logit2, logit3, logit4
