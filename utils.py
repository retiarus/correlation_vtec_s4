import os
import pdb

import numpy as np

from scipy.signal import convolve, gaussian, savgol_filter
from sklearn.metrics import mean_squared_error

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
    

def give_error(real, predict, verbose=True):
    mse = mean_squared_error(real, predict)
    tse = total_squared_error(real, predict)
    me = max_error(real, predict)
    re = relative_error(real, predict)
    
    if verbose:
        print("O erro quadrático médio foi: %f" %mse)
        print("O erro quadrático total foi: %f" %tse)
        print("O maior erro por previsão foi: %f" %me)
        print("O erro relativo foi %f%%" %re)
    
    return (mse, tse, me, re)