# predict.py
# design by Andy
# time: 2019/09/19
# to predict 25 year`s climate in the future
# we use the previous month`s climate to predict next month`s climate by RNN(Recurrent Neural Network)

import torch
import numpy as np
import pandas as pd

from model import RNN
from data_filter import get_data, get_global_temp
from config import PARAMS, DEVICE

epochs, look_back, hidden_layer, output_size, num_layers, lr, input_feature_size, out_feature_size = PARAMS
datas, min, max = get_global_temp(predict=True)
if len(datas.shape) == 1:
    input_feature_size = 1
else:
    input_feature_size = datas.shape[1]

datas = torch.from_numpy(datas).to(DEVICE).float()
datas = datas.reshape(1, input_feature_size, look_back)

# load model
model = RNN(look_back, hidden_layer, output_size, num_layers).to(DEVICE)
model.load_state_dict(torch.load('./model.pkl'))

predict_result = []
for i in range(25):
    result = model(datas)

    datas[:, :, 0:look_back-1] = datas[:, :, 1:look_back]
    datas[:, :, look_back-1] = result[:, :, 0]
    result = result.view(-1, input_feature_size).data.cpu().numpy()
    predict_result.append(result)

predict_result = np.concatenate(predict_result, 0)
predict_result = predict_result * (max - min) + min

predict_result = pd.DataFrame(predict_result)
predict_result.to_csv('./predict.csv')






