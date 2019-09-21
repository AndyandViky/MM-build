# read .nc file
# design by Andy
# time: 2019/09/19

import os
import copy
import netCDF4, urllib
import pylab, matplotlib
import numpy as np
import pandas as pd

from netCDF4 import Dataset
from config import ROOT_DIR
from utils import get_years_mean
# from opendap import opendap # OpenEarthTools module, see above that makes pypdap quack like netCDF4


def write_nc2csv(root, file_path="datasets/sst.mean.nc"):

    DATA_PATH = os.path.join(root, file_path)

    data = Dataset(DATA_PATH)

    keys = data.variables.keys()


    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    time = data.variables['time'][:]
    sst = data.variables['sst'][:]

    sst_temp = None

    dict_data = {}
    dict_data['time'] = time
    temp_array = np.zeros(len(time))
    temp_array[0:lat.size] = lat
    dict_data['lat'] = temp_array
    temp_array = temp_array * 0
    temp_array[0:lon.size] = lon
    dict_data['lon'] = temp_array
    temp_array = temp_array * 0
    temp_array[0] = len(time)
    dict_data['size'] = temp_array
    temp_data = pd.DataFrame(dict_data)
    temp_data.to_csv(os.path.join(ROOT_DIR, 'datasets/sst_mean_att.csv'))

    for i in range(180):
        sst1 = np.asarray(sst[:, :, i])
        sst1[sst1 == 32766.0] = np.nan
        if sst_temp is None:
            sst_temp = sst1
        else:
            sst_temp = np.vstack((sst_temp, sst1))

    fram_data = pd.DataFrame(sst_temp)
    fram_data.to_csv(os.path.join(ROOT_DIR, 'datasets/sst_mean.csv'))


def calculate_month_and_years_mean(size, lat_size, lon_size, root, filename, weight, month_fname, years_fname):
    """
    calculate average data, the data is 3-D
    the data was handle by write_nc2csv function
    :return:
    """
    data = pd.read_csv(os.path.join(root, filename), index_col=[0])
    data = data.values

    temp_month = []
    for month in range(size):
        # calculate every month mean temp in sea
        count = 0
        lat_temp = 0
        for lat in range(lat_size):
            # lat_size个维度
            _mean = np.nanmean(data[lat*size + month])
            if not np.isnan(_mean):
                lat_temp += _mean
                count += 1
        if count == 0:
            temp_month.append(np.nan)
        else:
            temp_month.append(lat_temp / count + weight)

    # calculate years
    temp_month = np.asarray(temp_month)
    years_average = get_years_mean(size, temp_month)

    years_average = pd.DataFrame(years_average)
    temp_month = pd.DataFrame(temp_month)
    temp_month.to_csv(os.path.join(ROOT_DIR, month_fname))
    years_average.to_csv(os.path.join(ROOT_DIR, years_fname))


# calculate_month_and_years_mean(2636, 180, 90, ROOT_DIR,
#                                'datasets/sst_mean.csv', 1.2, 'datasets/sea_month.csv', 'datasets/sea_year.csv')


# calculate_month_and_years_mean(535, 73, 144, ROOT_DIR,
#                                'datasets/olr_mean.csv', 18.5, 'datasets/olr_month.csv', 'datasets/olr_year.csv')


def write_nc2csv_olr(root, file_path="datasets/olr.mon.mean.nc"):

    DATA_PATH = os.path.join(root, file_path)

    data = Dataset(DATA_PATH)

    keys = data.variables.keys()


    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    time = data.variables['time'][:]
    olr = data.variables['olr'][:]

    olr_temp = None

    dict_data = {}
    dict_data['time'] = time
    temp_array = np.zeros(len(time))
    temp_array[0:lat.size] = lat
    dict_data['lat'] = temp_array
    temp_array = temp_array * 0
    temp_array[0:lon.size] = lon
    dict_data['lon'] = temp_array
    temp_array = temp_array * 0
    temp_array[0] = len(time)
    dict_data['size'] = temp_array
    temp_data = pd.DataFrame(dict_data)
    temp_data.to_csv(os.path.join(ROOT_DIR, 'datasets/olr_mean_att.csv'))

    for i in range(olr.shape[2]):
        olr1 = np.asarray(olr[:, :, i])
        # olr1[olr1 == 32766.0] = np.nan
        if olr_temp is None:
            olr_temp = olr1
        else:
            olr_temp = np.vstack((olr_temp, olr1))

    fram_data = pd.DataFrame(olr_temp)
    fram_data.to_csv(os.path.join(ROOT_DIR, 'datasets/olr_mean.csv'))


# data = pd.read_csv(os.path.join(ROOT_DIR, 'datasets/olr_month.csv'), index_col=False)
# last = len(data)-1
# for i in range(55):
#     a = np.random.random()*400
#     data.append(data[last] + a)
#     last += 1
#
# data = pd.DataFrame(data)
# data.to_csv(os.path.join(ROOT_DIR, 'datasets/olr_month.csv'))
