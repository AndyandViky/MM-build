# data_filter.py
# design by Andy
# time: 2019/09/19

import os
import numpy as np
import pandas as pd

from config import DATA_PATH, PARAMS, ROOT_DIR
from utils import get_years_mean


def get_data(predict=False):
    areas = ['west', 'east', 'north', 'south']

    datas = None
    for area, i in enumerate(areas):

        data_dir = os.path.join(DATA_PATH, i)
        if os.path.exists(data_dir):
            for parent, dirnames, filename in os.walk(data_dir):
                for index, name in enumerate(filename):
                    if name.split(".")[1] == 'csv':
                        data_path = os.path.join(data_dir, name)

                        data = pd.read_excel(data_path)
                        data['city'] = np.ones(len(data), dtype=np.int64) * int(index)
                        data['area'] = np.ones(len(data), dtype=np.int64) * int(area)
                        if datas is None:
                            datas = data
                        else:
                            datas = datas.append(data)

    datas.reset_index(drop=True, inplace=True)
    # Feature Engineering

    datas.dropna(subset=['Min_Temp', 'Max_Temp'], inplace=True)
    datas.reset_index(drop=True, inplace=True)

    datas['Total_Rain'] = datas['Total_Rain'].fillna(datas['Total_Rain'].mean())

    datas['Total_Snow'] = datas['Total_Snow'].fillna(datas['Total_Snow'].mean())

    datas['Total_Precip'] = datas['Total_Precip'].fillna(datas['Total_Precip'].mean())

    datas['Snow_on_Grnd'] = datas['Snow_on_Grnd'].fillna(datas['Snow_on_Grnd'].mean())

    datas['Speed'] = datas['Speed'].apply(lambda x: np.float(x) if x != '<31' else 31)
    datas['Speed'] = datas['Speed'].fillna(datas['Speed'].median())
    datas['Speed'] = (datas['Speed']-datas['Speed'].min())/(datas['Speed'].max()-datas['Speed'].min())

    datas['Date'] = datas['Date'].astype(str)
    date = datas['Date']
    city = datas['city']
    area = datas['area']
    datas.drop(['Date', 'area', 'city'], axis=1, inplace=True)

    max = datas['Mean_Temp'].max()
    min = datas['Mean_Temp'].min()
    # regulization
    datas = (datas - datas.min()) / (datas.max() - datas.min())
    datas['city'] = city
    datas['area'] = area

    test = {}
    test['Mean_Temp'] = datas['Mean_Temp']
    # test['Max_Temp'] = datas['Max_Temp']
    # test['Speed'] = datas['Speed']
    test = pd.DataFrame(test).values

    look_back = PARAMS[1]
    if predict:
        return test[365-look_back: 365], min, max
    else:
        return test[0: 365]


def get_global_temp(predict=False, mean=False, type="T"):

    if type == 'T':
        data = pd.read_excel(os.path.join(DATA_PATH, 'global_temp.xlsx'))
        data = data['total'][50*12:]
    elif type == 'P':
        data = get_csv('datasets/precip_month.csv')
        data = data['0']
    else:
        return None

    sea = get_csv(fname='datasets/sea_year.csv')
    co2 = get_csv(fname='datasets/co2_mean.csv')

    min = data.min()
    max = data.max()

    years_data = get_years_mean(len(data), data.values)
    if mean:
        return years_data

    else:
        final_data = {}
        final_data['total'] = years_data
        final_data['sea'] = np.concatenate(sea.values, 0)
        final_data['co2'] = np.concatenate(co2.values, 0)
        final_data = pd.DataFrame(final_data)
        final_data = (final_data - final_data.min()) / (final_data.max() - final_data.min())

        look_back = PARAMS[1]
        if predict:
            return final_data.values[-1-look_back: -1], min, max
        else:
            return final_data.values, min, max


def get_csv(fname):
    return pd.read_csv(os.path.join(ROOT_DIR, fname), index_col=[0])






















