# train_rnn.py
# design by Andy
# time: 2019/09/19

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.dates as mdate

from data_filter import get_data, get_global_temp
from model import RNN
from config import PARAMS, DEVICE, ROOT_DIR
from utils import creat_dataset
from generate_picture import generate_base


epochs, look_back, hidden_layer, output_size, num_layers, lr, input_feature_size, out_feature_size = PARAMS
datas, min, max = get_global_temp()
if len(datas.shape) == 1:
    input_feature_size = 1
else:
    input_feature_size = datas.shape[1]

# data loader and prepare
dataX, dataY = creat_dataset(datas, look_back)
dataX = torch.from_numpy(dataX).to(DEVICE)
dataY = torch.from_numpy(dataY).to(DEVICE)

train_size = int(len(dataX)*0.7)

x_train = dataX[:train_size]
y_train = dataY[:train_size]

x_train = x_train.view(-1, input_feature_size, look_back)
y_train = y_train.view(-1, input_feature_size, output_size)

model = RNN(look_back, hidden_layer, output_size, num_layers).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=2.5*1e-5)
loss_func = nn.MSELoss().to(DEVICE)

for i in range(epochs):
    x = x_train.float()
    y = y_train.float()

    out = model(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 5 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(i+1, loss.item()))

# save model
torch.save(model.state_dict(), './model.pkl')

# test
dataX = dataX.view(-1, input_feature_size, look_back).float()

pred = model(dataX)

pred_test = pred.view(-1, input_feature_size).data.numpy()

# result
for i in range(out_feature_size):
    true_data = pred_test[:, i] * (max - min) + min
    true_real_data = dataY[:, i] * (max - min) + min
    date = pd.date_range('1802-01', '2019-08', freq='12M')

    plt.rcParams["font.family"] = 'Arial Unicode MS'
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(date, true_data, 'b', label='prediction')
    ax.plot(date, true_real_data, 'r', label='real')
    ax.legend(loc='best')
    ax.set_ylabel(u'测试结果（℃）')

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))  # 设置时间标签显示格式
    ax.xaxis.set_major_locator(mdate.YearLocator())

    plt.xticks(pd.date_range('1819-01', '2019-08', freq='480M'), rotation=45)
    plt.show()
    fig.savefig(os.path.join(ROOT_DIR, 'datasets/test.svg'), dpi=600)

