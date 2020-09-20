# model.py
# design by Andy
# time: 2019/09/19

import torch.nn as nn


# model prepare
class RNN(nn.Module):
    def __init__(self, input_size=2, hidden_layer=6, output_size=1, num_layers=2):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_layer, self.num_layers, dropout=0.2)
        self.out = nn.Linear(self.hidden_layer, self.output_size)

    def forward(self, x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))
        out = out.view(a, b, -1)
        return out