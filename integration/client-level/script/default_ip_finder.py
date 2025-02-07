import numpy as np
import statsmodels.api as sm

def tangent_based(arr_value):
    ip = [0, 0]
    lat_array = np.array(arr_value)

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


def calc_trapezoid_area(len1, len2, h):
    return (len1 + len2) * h / 2

def calc_area_going_right_v0(y_idx, x_values, y_values):
    sum_area = 0
    n = len(x_values)
    max_x = np.max(x_values)
    # calculate the area going right
    for idx in range(y_idx, n):
        len1 = max_x - x_values[idx]

        len2 = max_x - x_values[idx - 1]
        h = abs(y_values[idx] - y_values[idx - 1])
        curr_area = calc_trapezoid_area(len1, len2, h)
        sum_area += curr_area
    return int(sum_area)

def calc_area_going_down_v0(y_idx, x_values, y_values):
    sum_area = 0
    n = len(y_values)
    # calculate the area going down
    for idx in range(y_idx):
        len1 = y_values[idx]
        len2 = y_values[idx + 1]
        h = abs(x_values[idx + 1] - x_values[idx])
        sum_area += calc_trapezoid_area(len1, len2, h)
    return int(sum_area)

def calc_area_going_left(lat_array, ip_latency_percent):
    sum_area = 0
    percentiles = [x for x in range(1, 101, 1)]
    n = len(percentiles)
    # print("percentiles = " + str(percentiles))
    lat_percentiles = np.percentile(lat_array, percentiles)
    lat_percentiles = np.around(lat_percentiles, decimals=2)

    for idx in range(ip_latency_percent):
        len1 = lat_percentiles[idx]
        len2 = lat_percentiles[idx + 1]
        h = abs(1)
        sum_area += calc_trapezoid_area(len1, len2, h)
    return sum_area

def area_based(arr_x_value, x_max = None):
    lat_array = np.array(arr_x_value)
    if x_max is None:
        # x_max is 5x of the median
        x_max = np.median(lat_array) * 5

    # remove the inf
    lat_array = lat_array[~np.isinf(lat_array)]

    # generate 0, 0.1, 0.2, ..., 100
    # percentiles = [float(x/10) for x in range(1, 1001, 1)]
    percentiles = [x for x in range(1, 101, 1)]
    n = len(percentiles)
    # print("percentiles = " + str(percentiles))
    lat_percentiles = np.percentile(lat_array, percentiles)
    lat_percentiles = np.around(lat_percentiles, decimals=2)

    # cut the long tail that exceeds x_max (5x of median)
    n_truncate = 0
    for idx, lat in enumerate(lat_percentiles):
        if lat >= x_max:
            lat_percentiles[idx] = x_max
            n_truncate += 1
    
    y_max = x_max

    # generate the y axis that is normalized to x_max
    y_axis_mirror = []
    for i in percentiles:
        y_axis_mirror.append(i * y_max / 100)
    x_values = lat_percentiles

    min_diff = float("inf")
    ip_percent = 0    
    min_area_threshold = x_max

    for y_idx in percentiles[1:-1]:
        area_down = calc_area_going_down_v0(y_idx, x_values, y_axis_mirror)
        area_right = calc_area_going_right_v0(y_idx, x_values, y_axis_mirror)
        diff = abs(area_down - area_right)
        # if one of the area is < min_area_threshold, then it's not a good candidate
        # if diff < min_diff and area_down > min_area_threshold and area_right > min_area_threshold:
        if diff < min_diff:
            min_diff = diff
            ip_percent = y_idx
        # print(" y_idx = " + str(y_idx) + " area_down = " + str(area_down) + " area_right = " + str(area_right) + " diff = " + str(diff) + " sum = " + str(area_down + area_right))
        # abs diff
    ip_latency = lat_percentiles[ip_percent - 1]
    # print("x_max = " + str(x_max))
    # print("ip_percent = " + str(ip_percent))
    # print("ip_latency = " + str(ip_latency))
    # exit(0)
    return ip_latency, ip_percent
    