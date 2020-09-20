# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: config.py
@time: 2020/9/17 10:59
@desc: config.py
'''
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(DEVICE))
# Local directory of CypherCat API
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory for datasets
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')

