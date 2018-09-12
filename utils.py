import os

default_location = "."
local_tecmap_txt = os.path.join(default_location, 'data/tecmap_txt' )
local_s4 = os.path.join(default_location, 'data_s4')
local_s4_pre = os.path.join(default_location, 'data_s4_pre')

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