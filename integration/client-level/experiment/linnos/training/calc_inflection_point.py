#!/usr/bin/env python

import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_file', help='input file', type=str, required=True)
    args = parser.parse_args()
    # print("input file for caclulating IP : " + args.input_file)
    with open(args.input_file) as f:
        arr_value = []
        for line in f:
            tok = list(map(str.strip, line.split(",")))
            arr_value.append(float(tok[1]))
        ip = tangent_based(arr_value)
        # print("IP: " + str(ip[0]) + " " + str(ip[1]))
        # print("IP Percentile : " + str(ip[1]) + "%")
        print(int(ip[1]))
    


# ./calc_inflection_point.py /mnt/extra2/daniar/flashnet-integration/data/grouping_2_traces_v9.per_3mins.for_nvme_12_n_13/alibaba.per_3mins.iops_p100.alibaba_9086.35/alibaba.per_3mins.iops_p100.alibaba_9086.35/modified.rerate_0.50...modified.rerate_4.00/nvme12n1...nvme13n1/baseline/trace_1.trace
