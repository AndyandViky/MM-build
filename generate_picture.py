#coding:utf-8
# generate_picture.py
# design by Andy
# time: 2019/09/19
# generate all the picture or diagram

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdate

from config import ROOT_DIR
from data_filter import get_global_temp, get_csv


def generate_base(data, dates, begin, title, label, freq='480M'):

    date = pd.date_range(dates[0], dates[1], freq='12M')

    plt.rcParams["font.family"] = 'Arial Unicode MS'
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(date, data, 'b', label=label)
    ax.legend(loc='best')
    ax.set_ylabel(title)

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))  # 设置时间标签显示格式
    ax.xaxis.set_major_locator(mdate.YearLocator())

    plt.xticks(pd.date_range(begin, '2019-08', freq=freq), rotation=45)
    plt.show()
    fig.savefig(os.path.join(ROOT_DIR, 'datasets/{}.svg'.format(label)), dpi=600)


def generate_earth_temp():

    earth_temp = get_global_temp(mean=True)
    generate_base(earth_temp, ('1800-01', '2019-08'), '1819-01', u'地表平均温度（℃）', 'earth_temp')


def generate_sea_temp():
    sea_temp = get_csv(fname='datasets/sea_year.csv')

    generate_base(sea_temp, ('1800-01', '2019-08'), '1819-01', u'海洋平均温度（℃）', 'sea_temp')


def generate_olr():
    sea_temp = get_csv(fname='datasets/olr_year.csv')

    generate_base(sea_temp, ('1975-01', '2019-08'), '1974-01', u'全球放热量（W/m^2）', 'olr_temp', '48M')


def generate_co2():
    co2_data = get_csv(fname='datasets/co2_mean.csv')

    generate_base(co2_data, ('1800-01', '2019-08'), '1819-01', u'全球平均二氧化碳排放量（千万吨）', 'co2_data', '480M')


def generate_loss():
    total_loss = pd.read_csv(os.path.join(ROOT_DIR, 'datasets/total_loss1.csv'), index_col=[0])
    plt.rcParams["font.family"] = 'Arial Unicode MS'
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.array(total_loss), 'b', label='loss')
    ax.set_ylabel(u'训练过程')
    plt.xticks(rotation=45)
    plt.show()
    fig.savefig(os.path.join(ROOT_DIR, 'datasets/train_loss.svg'), dpi=600)


def generate_predict():
    predict = get_csv('predict_temp.csv')
    test_data = get_csv('datasets/test_temp.csv')
    real_data = test_data['real_data']
    predict_data = test_data['predict_data']

    predict_data = predict_data.append(predict['0'])
    predict_data.reset_index()

    date = pd.date_range('1802-01', '2019-08', freq='12M')
    date1 = pd.date_range('1802-01', '2044-08', freq='12M')

    plt.rcParams["font.family"] = 'Arial Unicode MS'
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(date1, predict_data, 'b', label='prediction')
    ax.plot(date, real_data, 'r', label='real')
    ax.legend(loc='best')
    ax.set_ylabel(u'预测结果（℃）')

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))  # 设置时间标签显示格式
    ax.xaxis.set_major_locator(mdate.YearLocator())

    plt.xticks(pd.date_range('1804-01', '2044-08', freq='480M'), rotation=45)
    plt.show()
    fig.savefig(os.path.join(ROOT_DIR, 'datasets/predict.svg'), dpi=600)


def generate_predict_diagram():
    test_data = get_csv('datasets/test_temp.csv')
    test_data['error'] = test_data['real_data'] - test_data['predict_data']
    test_data.to_csv(os.path.join(ROOT_DIR, 'datasets/predict_error.csv'))


def generate_precip():
    precip_data = pd.read_csv(os.path.join(ROOT_DIR, 'datasets/precip_year.csv'), index_col=[0])

    generate_base(precip_data, ('1800-01', '2019-06'), '1819-01', u'全球降水量（mm）', 'precip_data', '480M')


# generate_precip()
# generate_predict_diagram()
# generate_predict()
# generate_earth_temp()
# generate_sea_temp()
# generate_olr()
# generate_co2()
# generate_loss()