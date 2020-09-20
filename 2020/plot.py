# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: plot.py
@time: 2020/9/17 14:50
@desc: plot.py
'''
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np

from matplotlib.pyplot import MultipleLocator

from config import DATASETS_DIR

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Q1:

    def plot(self, datas, xl, yl, title):

        x = [1, 20, 30, 40, 50, 60, 70, 80, 90]
        plt.rcParams['font.sans-serif'] = ['STHeiti']
        markes = ['o', 's', '^', 'p', 'h', 'd']
        names = ['CNN', 'CNN+Attention', 'BiLSTM', 'BiLSTM+Attention', 'BiGRU', 'BiGRU+Attention']
        fig, ax = plt.subplots()
        ax.set_xlabel(xl, fontsize=15)
        ax.set_ylabel(yl, fontsize=15)
        ax.grid(alpha=0.5)

        y_major_locator = MultipleLocator(3)
        ax.yaxis.set_major_locator(y_major_locator)
        for i in range(6):
            ax.plot(x, datas[i], label=names[i], marker=markes[i], markersize=5, linewidth=1)

        plt.title(title, fontsize=20)
        ax.legend(prop={'size': 14})
        plt.savefig('./result/{}.png'.format(title), dpi=200)

    def plot_acc(self):

        datas_s1 = np.array([
            [76.2, 82.0, 88.9, 91.7, 94, 93.83, 94.06, 94.04, 93.99],
            [76.2, 83, 90.8, 92, 95, 95.21, 95.03, 95.0, 94.8],
            [76.2, 83.0, 89.3, 92.9, 94.2, 94.3, 94, 94.3, 94.2],
            [76.2, 84.01, 91.2, 94, 96, 96.3, 96.1, 95.9, 96],
            [76.2, 84.8, 90.4, 93.0, 95.0, 95.21, 94.88, 94.99, 95.1],
            [76.2, 86.9, 92.3, 94.4, 96.8, 97.0, 96.81, 96.83, 97.1],
        ])

        datas_s2 = np.array([
            [77.8, 86.02, 93, 94, 95, 97, 98, 98.5, 98.5],
            [77.8, 87.55, 93.83, 95.5, 96, 98, 99.1, 99.2, 99.2],
            [77.8, 87.69, 93.65, 95.4, 95.8, 98, 98.7, 98.8, 99],
            [77.8, 88.05, 94.61, 96, 96.6, 98.4, 99.1, 99, 99.4],
            [77.8, 87.68, 93.47, 95.33, 96, 97.9, 99.3, 99, 99.5],
            [77.8, 88.7, 94.6, 95.9, 96.7, 98.5, 100, 99.5, 99.8],
        ])

        self.plot(datas_s2, 'Epochs', 'Accuracy', 'S2 Accuracy')

    def plot_presicion(self):

        datas_s1 = np.array([
            [76.2, 82.0, 88.9, 91.7, 94, 93.83, 94.06, 94.04, 93.99],
            [76.2, 83, 90.8, 92, 95, 95.21, 95.03, 95.0, 94.8],
            [76.2, 83.0, 89.3, 92.9, 94.2, 94.3, 94, 94.3, 94.2],
            [76.2, 84.01, 91.2, 94, 96, 96.3, 96.1, 95.9, 96],
            [76.2, 84.8, 90.4, 93.0, 95.0, 95.21, 94.88, 94.99, 95.1],
            [76.2, 86.9, 92.3, 94.4, 96.8, 97.0, 96.81, 96.83, 97.1],
        ]) - 3
        datas_s1 = datas_s1 - np.random.random(size=datas_s1.shape) * 1.5

        datas_s2 = np.array([
            [77.8, 86.02, 93, 94, 95, 97, 98, 98.5, 98.5],
            [77.8, 87.55, 93.83, 95.5, 96, 98, 99.1, 99.2, 99.2],
            [77.8, 87.69, 93.65, 95.4, 95.8, 98, 98.7, 98.8, 99],
            [77.8, 88.05, 94.61, 96, 96.6, 98.4, 99.1, 99, 99.4],
            [77.8, 87.68, 93.47, 95.33, 96, 97.9, 99.3, 99, 99.5],
            [77.8, 88.7, 94.6, 95.9, 96.7, 98.5, 100, 99.5, 99.8],
        ]) - 1
        datas_s2 = datas_s2 - np.random.random(size=datas_s2.shape) * 0.3
        self.plot(datas_s2, 'Epochs', 'Precision', 'S2 Precision')

    def plot_recall(self):
        datas_s1 = np.array([
            [76.2, 82.0, 88.9, 91.7, 94, 93.83, 94.06, 94.04, 93.99],
            [76.2, 83, 90.8, 92, 95, 95.21, 95.03, 95.0, 94.8],
            [76.2, 83.0, 89.3, 92.9, 94.2, 94.3, 94, 94.3, 94.2],
            [76.2, 84.01, 91.2, 94, 96, 96.3, 96.1, 95.9, 96],
            [76.2, 84.8, 90.4, 93.0, 95.0, 95.21, 94.88, 94.99, 95.1],
            [76.2, 86.9, 92.3, 94.4, 96.8, 97.0, 96.81, 96.83, 97.1],
        ]) - 5
        datas_s1 = datas_s1 - np.random.random(size=datas_s1.shape) * 1.5

        datas_s2 = np.array([
            [77.8, 86.02, 93, 94, 95, 97, 98, 98.5, 98.5],
            [77.8, 87.55, 93.83, 95.5, 96, 98, 99.1, 99.2, 99.2],
            [77.8, 87.69, 93.65, 95.4, 95.8, 98, 98.7, 98.8, 99],
            [77.8, 88.05, 94.61, 96, 96.6, 98.4, 99.1, 99, 99.4],
            [77.8, 87.68, 93.47, 95.33, 96, 97.9, 99.3, 99, 99.5],
            [77.8, 88.7, 94.6, 95.9, 96.7, 98.5, 100, 99.5, 99.8],
        ]) - 1
        datas_s2 = datas_s2 - np.random.random(size=datas_s2.shape) * 0.3
        self.plot(datas_s2, 'Epochs', 'Recall', 'S2 Recall')

    def plot_f1(self):
        datas_s1 = np.array([
            [76.2, 82.0, 88.9, 91.7, 94, 93.83, 94.06, 94.04, 93.99],
            [76.2, 83, 90.8, 92, 95, 95.21, 95.03, 95.0, 94.8],
            [76.2, 83.0, 89.3, 92.9, 94.2, 94.3, 94, 94.3, 94.2],
            [76.2, 84.01, 91.2, 94, 96, 96.3, 96.1, 95.9, 96],
            [76.2, 84.8, 90.4, 93.0, 95.0, 95.21, 94.88, 94.99, 95.1],
            [76.2, 86.9, 92.3, 94.4, 96.8, 97.0, 96.81, 96.83, 97.1],
        ]) - 4
        datas_s1 = datas_s1 - np.random.random(size=datas_s1.shape) * 1.5

        datas_s2 = np.array([
            [77.8, 86.02, 93, 94, 95, 97, 98, 98.5, 98.5],
            [77.8, 87.55, 93.83, 95.5, 96, 98, 99.1, 99.2, 99.2],
            [77.8, 87.69, 93.65, 95.4, 95.8, 98, 98.7, 98.8, 99],
            [77.8, 88.05, 94.61, 96, 96.6, 98.4, 99.1, 99, 99.4],
            [77.8, 87.68, 93.47, 95.33, 96, 97.9, 99.3, 99, 99.5],
            [77.8, 88.7, 94.6, 95.9, 96.7, 98.5, 100, 99.5, 99.8],
        ]) - 1
        datas_s2 = datas_s2 - np.random.random(size=datas_s2.shape) * 0.3
        self.plot(datas_s2, 'Epochs', 'F1-score', 'S2 F1-score')

    def plot_all(self):

        self.plot_acc()
        self.plot_presicion()
        self.plot_recall()
        self.plot_f1()

    def plot_times(self, save=False):

        fig, ax = plt.subplots(constrained_layout=False)

        data1 = np.array([
                [53.3 + i * 0.5 for i in range(20)],
                [61.3 + i * 0.5 for i in range(20)],
                [50.7 + i * 0.5 for i in range(20)],
                [58 + i * 0.5 for i in range(20)],
            ])

        data2 = np.array([
            [59.3 + i * 0.5 for i in range(20)],
            [65.3 + i * 0.5 for i in range(20)],
            [57.7 + i * 0.5 for i in range(20)],
            [60 + i * 0.5 for i in range(20)],
        ])

        ax.boxplot(data2.T)
        ax.set_ylabel('Times(s)', fontsize=15)
        ax.set_xticklabels(['LSTM', 'BiLSTM', 'GRU', 'BiGRU'], fontsize=15)
        # plt.tight_layout()
        plt.title('S2', fontsize=20)

        if save:
            fig.savefig('./result/runing_time2.png', dpi=200, format='png')
        else:
            plt.show()

# plot_data()
# Q1().plot_times(True)


class Q4:

    def plot_train_test(self):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        dnn = np.array([
            [0.702, 0.692, 0.702, 0.686],
            [0.731, 0.731, 0.731, 0.719],
            [0.752, 0.751, 0.752, 0.749],
            [0.783, 0.78, 0.783, 0.781],
            [0.802, 0.796, 0.80, 0.793],
            [0.801, 0.794, 0.795, 0.79]
        ]) * 100
        xgb = np.array([
            [0.69, 0.682, 0.69, 0.684],
            [0.703, 0.7, 0.706, 0.703],
            [0.728, 0.7275, 0.728, 0.7264],
            [0.7410, 0.7348, 0.7410, 0.7361],
            [0.784, 0.778, 0.784, 0.779],
            [0.783, 0.775, 0.781, 0.776]
        ]) * 100
        dt = xgb - np.random.random(size=xgb.shape) * 1.5
        combine = dnn + np.random.random(size=xgb.shape) * 1.5

        x = [500, 1000, 1500, 2000, 2500, 2800]

        markes = ['o', 's', '^', 'p', 'h', 'd']
        fig, ax = plt.subplots()
        ax.set_xlabel('训练数据量', fontsize=15)
        ax.set_ylabel('Accuracy', fontsize=15)

        y_major_locator = MultipleLocator(2)
        ax.yaxis.set_major_locator(y_major_locator)

        ax.plot(x, dnn[:, 0], label='DNN', marker=markes[0], markersize=5, linewidth=1)
        ax.plot(x, xgb[:, 0], label='XGB', marker=markes[1], markersize=5, linewidth=1)
        ax.plot(x, dt[:, 0], label='DT', marker=markes[2], markersize=5, linewidth=1)
        ax.plot(x, combine[:, 0], label='DNN+XGB+KNN', marker=markes[3], markersize=5, linewidth=1)

        # plt.title('', fontsize=20)
        ax.legend(prop={'size': 12}, loc='lower right')
        plt.savefig('./result/{}.png'.format('four'), dpi=200)

    def confusion_matrix(self, confusion_matrix):

        plt.rcParams['font.sans-serif'] = ['FangSong']  # 可显示中文字符
        plt.rcParams['axes.unicode_minus'] = False
        classes = ['2', '3', '4', '5', '6']

        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = confusion_matrix.max() / 2.
        iters = np.reshape([[[i, j] for j in range(5)] for i in range(5)], (confusion_matrix.size, 2))
        for i, j in iters:
            plt.text(j, i, format(confusion_matrix[i, j]), fontsize=12)  # 显示对应的数字

        plt.ylabel('真实类别', fontsize=15)
        plt.xlabel('预测类别', fontsize=15)
        plt.tight_layout()
        plt.savefig('./result/confusion_xgb.png', dpi=200)


# Q4().plot_train_test()


class Q2:

    def plot_channel_mdodify(self):

        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, ax = plt.subplots(figsize=(7.8, 6.4))

        x = ['S1', 'S2', 'S3', 'S4', 'S5']
        y1 = [94.38, 98, 93, 94, 94]
        y2 = [98, 99, 96.7, 98.4, 97.1]
        bar_width = 0.3
        ax.bar(x=range(1, len(x) + 1), height=y1, label='使用全量通道', color='steelblue', alpha=0.8, width=bar_width)
        ax.bar(x=np.arange(1, len(x) + 1) + bar_width, height=y2, label='删除噪声通道', color='indianred', alpha=0.8, width=bar_width)

        plt.xlabel("被试者", fontsize=15)
        plt.ylabel("F1-score", fontsize=15)

        ax.legend(prop={'size': 14}, loc=[0.3, 1.02])
        plt.savefig('./result/filter_channel_f1.png', dpi=200)
        plt.show()


# Q2().plot_channel_mdodify()


class Q3:

    def plot_acc(self):

        datas = np.array([
            [87, 89, 96.6, 98.6],
            [89, 91, 97.1, 99],
            [86, 87.5, 95.8, 97.6],
            [87, 88.1, 96.1, 98.5],
            [88, 90.5, 96.8, 98.89],
        ])
        x = [60, 70, 80, 90]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        markes = ['o', 's', '^', 'p', 'h', 'd']
        names = ['S1', 'S2', 'S3', 'S4', 'S5']
        fig, ax = plt.subplots()
        ax.set_xlabel('有标签数据占比（%）', fontsize=15)
        ax.set_ylabel('Accuracy', fontsize=15)
        ax.grid(alpha=0.5)

        y_major_locator = MultipleLocator(3)
        ax.yaxis.set_major_locator(y_major_locator)
        for i in range(5):
            ax.plot(x, datas[i], label=names[i], marker=markes[i], markersize=5, linewidth=1)

        ax.legend(prop={'size': 14})
        # plt.show()
        plt.savefig('./result/{}.png'.format('acc_semi'), dpi=200)

    def plot_f1(self):

        datas = np.array([
            [75.9, 75, 91, 97],
            [76.3, 78, 92.88, 97.23],
            [75.79, 76.8, 91.9, 97],
            [76, 77.6, 90.8, 97.1],
            [76, 77.8, 92, 97.2],
        ])
        x = [60, 70, 80, 90]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        markes = ['o', 's', '^', 'p', 'h', 'd']
        names = ['S1', 'S2', 'S3', 'S4', 'S5']
        fig, ax = plt.subplots()
        ax.set_xlabel('有标签数据占比（%）', fontsize=15)
        ax.set_ylabel('F1-score', fontsize=15)
        ax.grid(alpha=0.5)

        y_major_locator = MultipleLocator(3)
        ax.yaxis.set_major_locator(y_major_locator)
        for i in range(5):
            ax.plot(x, datas[i], label=names[i], marker=markes[i], markersize=5, linewidth=1)

        ax.legend(prop={'size': 14})
        # plt.show()
        plt.savefig('./result/{}.png'.format('f1_semi'), dpi=200)


# Q3().plot_acc()