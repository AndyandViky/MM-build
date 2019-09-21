# utils.py
# design by Andy
# time: 2019/09/19

import numpy as np


def creat_dataset(dataset, look_back):
    """
    create a dataset for LSTM
    :param dataset:
    :param look_back:
    :return:
    """
    data_x = []
    data_y = []
    for i in range(len(dataset)-look_back):
        data_x.append(dataset[i:i+look_back])
        data_y.append(dataset[i+look_back])
    return np.asarray(data_x), np.asarray(data_y)


def get_years_mean(size, data, change_type='M'):
    """
    change data to 'years mean' by 'months mean' or other
    :param size:
    :param data:
    :parm change_type: choose M or D, default is M
    :return:
    """
    assert isinstance(data, np.ndarray) and size > 0

    if change_type == 'M':
        divided = 12
    elif change_type == 'D':
        divided = 30 # every month to 30 days
    else:
        return None
    last_years_month = size % divided
    total_years = int((size - last_years_month) / divided) + 1

    years_average = []
    for i in range(total_years):
        if i == total_years - 1:
            years_average.append(np.nanmean(data[i * divided: i * divided + last_years_month]))
        else:
            years_average.append(np.nanmean(data[i * divided: i * divided + divided]))

    return np.array(years_average)

