import datetime
import os
import sys
import pdb

import numpy as np
import pandas as pd

from scipy.signal import convolve, gaussian, savgol_filter
from sklearn.metrics import mean_squared_error

curr_dir = os.path.abspath(os.path.curdir)
os.chdir("/home/peregrinus/suntime")
import suntime.__main__ as suntime
os.chdir(curr_dir)

default_location = "."
local_tecmap_txt = os.path.join(default_location, 'data/tecmap_txt')
local_data = os.path.join(default_location, 'data')
local_s4 = os.path.join(default_location, 'data_s4')
local_s4_pre = os.path.join(default_location, 'data_s4_pre')
window = 9

group_1 = ['bhz', '30', '25', 'sj2']

def extract_vtec(series):
    return [row.vtec for index, row in series.iterrows()]

def extract_index(series):
    return [index for index, row in series.iterrows()]
    
def i_longitude(value):
    if value <= 0.0:
        return int((value+100.0)*2.0)

def j_latitude(value):
    return int((value+60.0)*2.0)

def ij_par(value_i, value_j):
    return i_longitude(value_i), j_latitude(value_j)

def second_order_derivative_time(data):
    result = []
    size = len(data)
   
    for i in range(1, size -1):
        result.append(2*data[i] - data[i-1] - data[i+1])
        
    return result

def first_order_derivative_time(data):
    result = []
    size = len(data)
   
    for i in range(1, size):
        result.append(data[i] - data[i-1])
        
    return result

def smooth_signal(sig):
    window_gaussian = gaussian(window, std=1)
    window_gaussian = window_gaussian/window_gaussian.sum()
    par = window//2
    
    # aplication of the filters
    sig = savgol_filter(sig, window, 3)
    sig = convolve(sig, window_gaussian, mode='same')
    
    for i in range(0, par):
        sig[i] = np.nan
    for i in range(-1, -(par+1), -1):
        sig[i] = np.nan
    
    return sig

class Scale(object):
    def __init__(self, value_min, value_max):
        self._value_min = value_min
        self._value_max = value_max
        self._desvio = self._value_max - self._value_min
        
    def __call__(self, x_array):
        x = x_array.copy()
        x = (x - self._value_min)/self._desvio
        return x

def total_squared_error(real, predict):
    value = (real - predict)*(real - predict)
    value_sum = value.sum()
    return np.sqrt(value_sum)

def max_error(real, predict):
    value = np.abs(real - predict)
    return value.max()

def relative_error(real, predict):
    value = np.abs(real - predict)
    value /= np.abs(real)
    return value.mean()*100

def pseudo_confusion_matrix(real, predict):
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    cut_value = 0.2
    
    for r, p in zip(real, predict):
        if r <= cut_value and p <= cut_value:
            true_negative += 1
        elif r > cut_value and p > cut_value:
            true_positive += 1
        elif r > cut_value and p <= cut_value:
            false_negative += 1
        elif r <= cut_value and p > cut_value:
            false_positive += 1
            
    return (true_negative, true_positive, false_negative, false_positive)

def classification_metrics(true_negative, true_positive, false_negative, false_positive):
    total = true_negative \
            + true_positive \
            + false_negative \
            + false_positive
    pod = float(true_positive)/float(true_positive+false_negative)
    far = float(false_positive)/float(true_positive+false_positive)
    acc = float(true_negative + true_positive)/total
    kappa = float(true_negative*false_negative + true_positive*false_positive)/float(total*total)
    return (pod, far, acc, kappa)
    
def give_error(real, predict, verbose=True):
    mse = mean_squared_error(real, predict)
    tse = total_squared_error(real, predict)
    me = max_error(real, predict)
    re = relative_error(real, predict)
    true_negative, true_positive, false_negative, false_positive = pseudo_confusion_matrix(real, predict)
    pod, far, acc, kappa = classification_metrics(true_negative,
                                                  true_positive,
                                                  false_negative,
                                                  false_positive)
    
    if verbose:
        print("O erro quadrático médio foi: %f" %mse)
        print("O erro quadrático total foi: %f" %tse)
        print("O maior erro por previsão foi: %f" %me)
        print("O erro relativo foi: %f%%" %re)
        print("O número de verdadeiros negativos foi: %i" %true_negative)
        print("O número de verdadeiros positivos foi: %i" %true_positive)
        print("O número de falsos negativos foi: %i" %false_negative)
        print("O número de falsos positivos foi: %i" %false_positive)
        print("O POD foi: %f" %pod)
        print("O FAR foi: %f" %far)
        print("A ACC foi: %f" %acc)
        print("O kappa foi: %f" %kappa)
    
    return (mse, tse, me, re)

def to_datetime(date):
    ns = 1e-9
    date_int = date.astype(int)
    return datetime.datetime.utcfromtimestamp(date_int * ns)
    
def is_post_sunset_partial(date, lat, long):
    datetime_ref = to_datetime(date)
    date_date = datetime_ref.date()
    date_time = datetime_ref.time()
    
    sun = suntime.Sun(lat, long)
    ss = sun.get_sunset_time(date_date)
    
    ss_time = ss.time()
    
    if date_time >= ss_time:
        return 1.0
    else:
        return 0.0
    
def is_post_sunrise_partial(date, lat, long):
    datetime_ref = to_datetime(date)
    date_date = datetime_ref.date()
    date_time = datetime_ref.time()
    
    sun = suntime.Sun(lat, long)
    sr = sun.get_sunrise_time(date_date)
    
    sr_time = sr.time()
    
    if date_time >= sr_time:
        return 1.0
    else:
        return 0.0
    
def is_day_partial(date, lat, long):
    datetime_ref = to_datetime(date)
    date_date = datetime_ref.date()
    date_time = datetime_ref.time()
    
    sun = suntime.Sun(lat, long)
    sr = sun.get_sunrise_time(date_date)
    ss = sun.get_sunset_time(date_date)
    sr_time = sr.time()
    ss_time = ss.time()
    
    if sr_time <= date_time < ss_time:
        return 1.0
    else:
        return 0.0
    
def is_night_partial(date, lat, long):
    datetime_ref = to_datetime(date)
    date_date = datetime_ref.date()
    date_time = datetime_ref.time()
    
    sun = suntime.Sun(lat, long)
    ss = sun.get_sunset_time(date_date)
    ss_time = ss.time()
    
    if ss_time <= date_time <= datetime.time(23, 59, 59):
        return 1.0
    else:
        return 0.0
    
def is_dawn_partial(date, lat, long):
    datetime_ref = to_datetime(date)
    date_date = datetime_ref.date()
    date_time = datetime_ref.time()
    
    sun = suntime.Sun(lat, long)
    sr = sun.get_sunrise_time(date_date)
    sr_time = sr.time()
    
    if datetime.time(0, 0) <= date_time < sr_time:
        return 1.0
    else:
        return 0.0

def find_set_sunset(df, lat, long):
    set_of_days = df.resample('D').mean().index.values

    sun = suntime.Sun(lat, long)

    set_of_sunset = []
    for i in set_of_days:
        datetime_ref = to_datetime(i).date()
        ss = sun.get_sunset_time(datetime_ref)
        set_of_sunset.append(np.datetime64(ss))
        
    return set_of_sunset

def find_set_sunrise(df, lat, long):
    set_of_days = df.resample('D').mean().index.values

    sun = suntime.Sun(lat, long)

    set_of_sunrise = []
    for i in set_of_days:
        datetime_ref = to_datetime(i).date()
        sr = sun.get_sunrise_time(datetime_ref)
        set_of_sunrise.append(np.datetime64(sr))
        
    return set_of_sunrise

def location_station(station):
    df_station = pd.read_pickle('./data/df_station_sort_re.pkl')
    
    lat = df_station.loc[df_station['identificationstation'] == station]['latitude'].values[0]
    long = df_station.loc[df_station['identificationstation'] == station]['longitude'].values[0]
    
    return lat, long

def plot_sunrise_and_sunset(station, df, axes):
    lat, long = location_station(station)
    set_of_sunrise = find_set_sunrise(df, lat, long)
    set_of_sunset = find_set_sunset(df, lat, long)
    
    try:
        _ = iter(axes)
    except TypeError as te:
        aux = []
        aux.append(axes)
        axes = aux
    
    for ax_i in axes:
        for i in set_of_sunrise:
            ax_i.axvline(x=i, color='y')
        for i in set_of_sunset:
            ax_i.axvline(x=i, color='r')
        
    