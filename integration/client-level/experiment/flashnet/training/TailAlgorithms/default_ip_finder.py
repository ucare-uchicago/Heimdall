import numpy as np
import statsmodels.api as sm

def tangent_based(arr_value):
    ip = [0, 0]
    lat_array = np.array(arr_value)
    # lat_array = arr_value

    # remove the inf
    lat_array = lat_array[~np.isinf(lat_array)]

    lat_97 = np.percentile(lat_array, 97)
    lat_array = lat_array[lat_array <= lat_97]
    max_lat = np.max(lat_array)
    lat_array = np.divide(lat_array, max_lat)

    ecdf = sm.distributions.ECDF(lat_array)
    x = np.linspace(0, 1, num=10000)
    y = ecdf(x)

    t = y - x
    ip_idx = np.argmax(t)
    ip[0] = x[ip_idx]
    ip[1] = y[ip_idx]
    ip[0] = int(ip[0] * max_lat)
    ip[1] = (ip[1] * 0.97 * 100)
    return ip