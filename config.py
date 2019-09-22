# config.py
# design by Andy
# time: 2019/09/19

import os
import torch

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "datasets")

# epochs, look_back, hidden_layer, output_size, num_layers, lr, input_feature_size, out_feature_size
# PARAMS = (500, 2, 15, 1, 1, 1e-2, 1, 1)
PARAMS = (3000, 2, 30, 1, 1, 1e-2, 1, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
