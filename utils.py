import copy
import datetime
import os
import sys
import pdb

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sympy import FiniteSet

from scipy.signal import convolve, gaussian, savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from cartopy.feature import NaturalEarthFeature, BORDERS, LAND, COASTLINE, COLORS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

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

def pseudo_confusion_matrix(real, predict, cut_value=0.2):
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
    precission = float(true_positive)/float(true_positive+false_positive)
    recall = float(true_positive)/float(true_positive+false_negative)
    f1 = 2*precission*recall/(precission+recall)
    return (pod, far, acc, kappa, precission, recall, f1)
    
def give_error(real, predict, verbose=True, cut_value=0.2):
    mse = mean_squared_error(real, predict)
    tse = total_squared_error(real, predict)
    me = max_error(real, predict)
    re = relative_error(real, predict)
    true_negative, true_positive, false_negative, false_positive = pseudo_confusion_matrix(real, predict, 0.2)
    pod, far, acc, kappa, precission, recall, f1 = classification_metrics(true_negative,
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
        print("precission: %f" %precission)
        print("recal: %f" % recall)
        print("f1: %f" % f1)
        
    dict_error = {'mse': mse,
                  'tse': tse,
                  'me': me,
                  're': re,
                  'tn': true_negative,
                  'tp': true_positive,
                  'fn': false_negative,
                  'fp': false_positive,
                  'pod': pod,
                  'far' : far,
                  'acc': acc,
                  'kappa': kappa,
                  'precission': precission,
                  'recall': recall,
                  'f1': f1}
    
    return dict_error

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
        
def prepare_map(figsize=(12,12)):
    linestyles = dict([('solid',               (0, ())),
                       ('loosely dotted',      (0, (1, 10))),
                       ('dotted',              (0, (1, 5))),
                       ('densely dotted',      (0, (1, 1))),
                       ('loosely dashed',      (0, (5, 10))),
                       ('dashed',              (0, (5, 5))),
                       ('densely dashed',      (0, (5, 1))),
                       ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                       ('dashdotted',          (0, (3, 5, 1, 5))),
                       ('densely dashdotted',  (0, (3, 1, 1, 1))),
                       ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                       ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                       ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-90, -25, -50, 20])

    land = NaturalEarthFeature(category='physical',
                               name='land',
                               scale='50m',
                               facecolor='none')
    ax.add_feature(land,
                   edgecolor='black')

    states = NaturalEarthFeature(category='cultural',
                                 scale='50m',
                                 facecolor='none',
                                 name='admin_1_states_provinces_shp')
    ax.add_feature(states,
                   edgecolor='lightgray',
                   linestyle='-',
                   linewidth=1)

    countries = NaturalEarthFeature(scale='50m',
                                    category='cultural',
                                    facecolor='none',
                                    name='admin_0_countries',
                                    alpha=1)
    ax.add_feature(countries,
                   edgecolor='gray',
                   linestyle='-',
                   linewidth=1)

    gl = ax.gridlines(draw_labels=True,
                      linewidth=1,
                      color='blue',
                      alpha=0.2,
                      linestyle=linestyles['solid'])

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(nbins=19, steps=[1, 2, 5, 10])
    gl.ylocator = mticker.MaxNLocator(nbins=19, steps=[1, 2, 5, 10])

    return fig, ax
  
def generate_vm_vd(df, deltatime):
    size = len(df)
    vm = np.empty(size)
    vd = np.empty(size)
    vm[:] = np.nan
    vd[:] = np.nan
    se_vm = pd.Series(vm, index=df.index.values)
    se_vd = pd.Series(vd, index=df.index.values)

    day_begin = df[(df.state_day == 1.0) & (df.state_day_shift == 0.0)].index.values
    day_end = df[(df.state_day == 0.0) & (df.state_day_shift == 1.0)].index.values

    if day_begin[0] > day_end[0]:
        day_end = day_end[1:]
    if len(day_begin) > len(day_end):
        day_begin = day_begin[0:-1]

    # o preíodo de amostragem começa no final do dia
    # e se extende até uma hora após o por do sol
    day_begin = copy.deepcopy(day_end)
    day_end += deltatime

    # remove o intervalo final, se o extremo a direita dat_end[]
    # cair pra fora da série
    if day_end[-1] not in df.index.values:
        day_end = day_end[0:-1]
        day_begin = day_begin[0:-1]

    # combina os pares e cálcula o valor médio ao longo do intervalo
    for day_begin_i, day_end_i in zip(day_begin, day_end):
        se_vm[day_begin_i] = df[day_begin_i:day_end_i]['vtec'].mean()
        se_vd[day_begin_i] = df[day_begin_i:day_end_i]['vtec'].std()

    before_dif_nonnan_vm = np.nan
    before_dif_nonnan_vd = np.nan
    for idx_vm, value_vm, idx_vd, value_vd in zip(se_vm.index, se_vm, se_vd.index, se_vd):
        if not np.isnan(value_vm) and (value_vm != before_dif_nonnan_vm):
            before_dif_nonnan_vm = value_vm
        if np.isnan(value_vm):
            se_vm[idx_vm] = before_dif_nonnan_vm
        if not np.isnan(value_vd) and (value_vd != before_dif_nonnan_vd):
            before_dif_nonnan_vd = value_vd
        if np.isnan(value_vd):
            se_vd[idx_vd] = before_dif_nonnan_vd

    return se_vm, se_vd

def generate_and_avaliate_model(df, 
                                instances_set, 
                                target, 
                                file_to_save_model=None, 
                                model=None, 
                                save=None,
                                grid_search=False,
                                param_grid=None,
                                random_state=42,
                                simple=False,
                                cut_value=0.2):
    if model is None:
        # generate default model RandomForest, when model is not defined
        model = RandomForestRegressor
        
        # define parameters for grid_search
        # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # max_features = ['auto', 'sqrt']
        # max_depth = [int(x) for x in np.linspace(10, 180, num=11)]
        # max_depth.append(None)
        # min_samples_split = [2, 5, 10]
        # min_samples_leaf = [1, 2, 4]
        # bootstrap = [True, False]

        n_estimators = [int(x) for x in np.linspace(start=500, stop=1000, num=10)]
        max_depth = [int(x) for x in np.linspace(10, 100, num=11)]
        
        if grid_search:
           # param_grid = {'model__n_estimators': n_estimators,
           #               'model__max_features': max_features,
           #               'model__max_depth': max_depth,
           #               'model__min_samples_split': min_samples_split,
           #               'model__min_samples_leaf': min_samples_leaf,
           #               'model__bootstrap': bootstrap}
           #  
            param_grid = {'model__n_estimators': n_estimators,
                          'model__max_depth': max_depth} 
    # select data
    X = df[instances_set].values
    y = df['s4'].values
    
    size = len(X)
    last_element = size - size//10
    
    X_train = X[0:last_element]
    y_train = y[0:last_element] 
       
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('model', model()))
    pipeline = Pipeline(estimators)
    
    if param_grid is not None:
        clf = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           cv=10,
                           verbose=2,
                           n_jobs=-1,
                           scoring='neg_mean_squared_error')
      
        clf.fit(X_train, y_train)
        best_parameters_estimator = clf.best_estimator_.get_params()
        best_parameters_model = best_parameters['model'].get_params()
    
    # generate standardize transformation for (x,y)
    X_scaler = StandardScaler() # transformation for X
    X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)

    # generate final model
    if True:
        mod = model(**best_parameters_model)
    else:
        pass
    
    mod.fit(X_train, y_train)
    
    # use the final model to avaliate the error in a sample of the time series
    X_validate = X_scaler.transform(X[last_element:size+1])
    y_validate = y[last_element:size+1]
    
    index = df.index.values[last_element:size+1]
    df_aux = pd.DataFrame(index=index)
    df_aux['predito'] = mod.predict(X_validate)
    df_aux['real'] = y_validate

    print('Error for the time series sample:')
    dict_error = give_error(df_aux['real'].values, df_aux['predito'].values, cut_value=cut_value);

    # plot the time series predict against the real values
    ax = df_aux.plot(figsize=(18, 8));
    plt.xlabel('UT')

    lat, long = location_station('sj2')
    set_of_sunrise = find_set_sunrise(df_aux, lat, long)
    set_of_sunset = find_set_sunset(df_aux, lat, long)
    for i in set_of_sunrise:
        ax.axvline(x=i, color='y')
    for i in set_of_sunset[0:-1]:
        ax.axvline(x=i, color='r')

    plt.tight_layout()
    if save:
        plt.savefig(file_to_save_model, format='eps', dpi=1000)
 
    plt.show()

        
    return dict_error

class MySet():
    def __init__(self, name, list_of_elements):
        self.set = set(list_of_elements)
        self.name = name
            
    def __add__(self, other_set):
        aux = copy.deepcopy(self)
        aux.set = aux.set.union(other_set.set)
        aux.name = aux.name + ' + ' + other_set.name
        return aux
    
    def __sub__(self, other_set):
        aux = copy.deepcopy(self)
        aux.set = aux.set.difference(other_set.set)
        aux.name = aux.name + ' - ' + other_set.name
        return aux
    
    def __str__(self):
        return self.name