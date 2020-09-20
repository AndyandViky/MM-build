# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: question_three.py
@time: 2020/9/17 21:03
@desc: question_three.py
'''
import numpy as np
import os
import torch
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import DEVICE
from datasets import get_dataloader


def LPA():

    dataloader = get_dataloader(dataset_path='S1', train=True, full=True, extend=False)
    datas, origin_labels = dataloader.dataset.data, dataloader.dataset.label

    leng = 720 - 72
    labels = np.copy(origin_labels)
    Y = labels[leng:].copy()  # label转换之前的
    labels[leng:] = -1  # 标签重置，将标签为1的变为-1
    print('Unlabeled Number:', list(labels).count(-1))

    from question_one import Classifier as Encoder
    classifier = Encoder(150, 32, 2).to(DEVICE)
    classifier.load_state_dict(torch.load('./classifier.pk'))
    datas = classifier.attention((torch.tensor(datas, dtype=torch.float32).to(DEVICE))).data.cpu().numpy()
    datas = (datas - np.min(datas, axis=1, keepdims=True)) / (
                np.max(datas, axis=1, keepdims=True) - np.min(datas, axis=1, keepdims=True))

    # from sklearn import manifold
    # import matplotlib.pyplot as plt
    # '''t-SNE'''
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # X_tsne = tsne.fit_transform(datas)
    #
    # print("Org data dimension is {}.Embedded data dimension is {}".format(datas.shape[-1], X_tsne.shape[-1]))
    #
    # '''嵌入空间可视化'''
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure()
    #
    # for i in range(X_norm.shape[0]):
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(origin_labels[i]))
    # plt.xticks([])
    # plt.yticks([])
    # # plt.savefig('./result/tsne_90%.png', dpi=200)
    # plt.show()

    laber_prop_model = LabelPropagation(kernel='knn')
    laber_prop_model.fit(datas, labels)
    Y_pred = laber_prop_model.predict(datas)
    Y_pred = Y_pred[leng:]  # -1的那部分重新预测
    print(len(Y[Y == 1]), len(Y_pred[Y_pred == 1]))
    # Y = origin_labels
    print('acc: {}, p: {}, r: {}, f1: {}'.format(accuracy_score(Y, Y_pred), precision_score(Y, Y_pred),
                                                 recall_score(Y, Y_pred), f1_score(Y, Y_pred)))


def test():
    dataloader = get_dataloader(dataset_path='S1', train=True, full=True, extend=False)
    datas, origin_labels = dataloader.dataset.data, dataloader.dataset.label
    test_loader = get_dataloader(dataset_path='S1', test=True, batch_size=5000, shuffle=False)
    # testing
    t_data, t_label = next(iter(test_loader))
    t_data = t_data.numpy()
    # datas = np.vstack((datas, t_data))
    # origin_labels = np.concatenate((origin_labels, t_label.numpy()), 0)

    leng = 720 - 72
    labels = np.copy(origin_labels)
    Y = labels[leng:].copy()  # label转换之前的
    labels[leng:] = -1  # 标签重置，将标签为1的变为-1
    print('Unlabeled Number:', list(labels).count(-1))

    from question_one import Classifier as Encoder
    classifier = Encoder(150, 32, 2).to(DEVICE)
    classifier.load_state_dict(torch.load('./classifier.pk'))
    datas = classifier.attention((torch.tensor(datas, dtype=torch.float32).to(DEVICE))).data.cpu().numpy()
    datas = (datas - np.min(datas, axis=1, keepdims=True)) / (
            np.max(datas, axis=1, keepdims=True) - np.min(datas, axis=1, keepdims=True))

    t_data = classifier.attention((torch.tensor(t_data, dtype=torch.float32).to(DEVICE))).data.cpu().numpy()
    t_data = (t_data - np.min(t_data, axis=1, keepdims=True)) / (
            np.max(t_data, axis=1, keepdims=True) - np.min(t_data, axis=1, keepdims=True))

    laber_prop_model = LabelPropagation(kernel='knn')
    laber_prop_model.fit(datas, labels)
    Y_pred = laber_prop_model.predict(t_data)

    for i in range(10):
        s = Y_pred[i*60: (i+1) * 60].reshape((5, 12))
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
    # LPA()
    # main()
    # XGB()
    test()

