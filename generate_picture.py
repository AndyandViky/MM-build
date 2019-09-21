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


# generate_earth_temp()
# generate_sea_temp()
# generate_olr()
# generate_co2()
