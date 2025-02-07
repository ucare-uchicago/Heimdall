#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd 
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

sys.path.insert(0, "../script/")
import pattern_checker
import default_ip_finder

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    # print("===== output file : " + filePath)

def get_algo_dirs(trace_dir):
    # list all dirs in this path
    algo_dirs = []
    algo_names = []
    for path in os.listdir(trace_dir):
        path = os.path.join(trace_dir, path)
        if os.path.isdir(path):
            if path.endswith("flashnet_norm"):
                # skipping flashnet_norm for now
                continue
            algo_dirs.append(path)
            algo_names.append(os.path.basename(path))
    return algo_dirs, algo_names

def is_all_algo_analyzed(arr_algo_dirs):
    # check if all algo dirs have the CDF generated
    for algo_dir in arr_algo_dirs:
        if not os.path.exists(os.path.join(algo_dir, "latency_characteristic.stats")):
            return False
    return True

def read_replayed_file(input_file):
    df = pd.read_csv(input_file, header=None, sep=',')
    # Make sure it has 7 columns
    assert df.shape[1] >= 7

    # remove columns index 7th ...
    df = df.iloc[:, :7]

    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)
    df.columns = ["ts_record","latency","io_type","size","offset","ts_submit","size_after_replay"]

    # filter: remove io that doesn't executed properly (can't read/write all bytes)
    df = df[df['size'] == df['size_after_replay']]
    return df

def get_per_trace_latency(algo_dir):
    arr_read_latency = []
    arr_trace_names = []
    traces_in_this_dir = [trace for trace in os.listdir(algo_dir) if (os.path.isfile(os.path.join(algo_dir, trace)) and str(trace).endswith('.trace'))]
    for trace_name in traces_in_this_dir:
        input_path = os.path.join(algo_dir, trace_name)
        filename = str(Path(trace_name).with_suffix('') ) # remove .trace extension
        df = read_replayed_file(input_path)

        # Only draw CDF for read IOs!!!
        df = df[df['io_type'] == 1]
        arr_read_latency += list(df['latency'])
        arr_trace_names.append(filename)
    return arr_read_latency, arr_trace_names

def get_percentiles(aggregated_latencies, N=100):
    # get p1,p2, ..., p100 of the latency
    divider = N/100
    percentiles = [x/divider for x in range(1, N+1)]
    lat_percentiles = np.percentile(aggregated_latencies, percentiles)
    return lat_percentiles
    # lat_percentiles = np.around(lat_percentiles, decimals=2)

def start_process_per_algo(algo_dir):
    latency_stats_str = []
    # get all latency 
    arr_read_latency, arr_trace_names = get_per_trace_latency(algo_dir)

    # Get the common characteristics 
    try:
        latency_stats_str.append("count = " + str(len(arr_read_latency)) + " IOs")
        latency_stats_str.append("avg = " + str(np.mean(arr_read_latency)) + " us")
        latency_stats_str.append("std = " + str(np.std(arr_read_latency)) + " us")
        latency_stats_str.append("median = " + str(np.median(arr_read_latency)) + " us")
        latency_stats_str.append("max = " + str(np.max(arr_read_latency)) + " us")
        latency_stats_str.append("min = " + str(np.min(arr_read_latency)) + " us")

        # Get 10000 percentile
        n_percentiles = 10000
        arr_10000_percentile = get_percentiles(arr_read_latency, N=n_percentiles)
        divisor = n_percentiles/100
        for idx, lat in enumerate(arr_10000_percentile):
            latency_stats_str.append("p" + str(round(((idx+1)/divisor), 2)) + " = " + str(lat) + " us")
    except:
        pass

    # write the stats to file
    output_path = os.path.join(algo_dir, "latency_characteristic.stats")
    write_stats(output_path, "\n".join(latency_stats_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-trace_dir", help="File path to the trace sections description ", type=str)
    parser.add_argument("-trace_dirs", help="Arr of file path to the trace sections description", nargs='+',type=str)
    parser.add_argument("-reverse", help="Reverse the order of the traces", action='store_true')
    parser.add_argument("-resume", help="Only run the analysis if there is newly run algorithm", action='store_true')
    parser.add_argument("-random", help="Randomize the order of the traces", action='store_true')
    args = parser.parse_args()
    if (not (args.trace_dir or args.trace_dirs)):
        print("    ERROR: You must provide these arguments: -trace_dir <the trace dir> or -trace_dirs <the array of trace dirs")
        exit(-1)
        
    trace_dirs = []
    if args.trace_dirs:
        trace_dirs += args.trace_dirs
    elif args.trace_dir:
        trace_dirs.append(args.trace_dir)
    print("trace_paths = " + str(trace_dirs))

    if args.reverse:
        trace_dirs = trace_dirs[::-1]
        print("Reversed the order of the trace dirs")

    if args.random:
        np.random.shuffle(trace_dirs)
        print("Randomized the order of the trace dirs")
        
    for idx, trace_dir in enumerate(trace_dirs):
        print("\n" + str(idx + 1) + ". (out of " + str(len(trace_dirs))+ ") Processing " + str(trace_dir))
        stats = []
        # this is the dir with dev1...dev2
        arr_algo_dirs, arr_algo_names = get_algo_dirs(trace_dir)
        per_algo_latencies = {}
        per_algo_100_percentiles = {}
        
        if args.resume:
            # check if all algo dirs have the CDF generated
            if is_all_algo_analyzed(arr_algo_dirs):
                print("===== Existing stats are already created = " + str(trace_dir))
                continue
        
        for algo_dir in arr_algo_dirs:
            algo_name = os.path.basename(algo_dir)
            print("    Processing " + algo_name)

            start_process_per_algo(algo_dir)
            