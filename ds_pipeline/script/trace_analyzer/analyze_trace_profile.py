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
from PIL import Image

# How to run:
def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(df, filePath, has_header=True):
    # The raw (replayed) traces don't have the header, so when we
    # write the filtered version, it must be consistent with the original
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

def plot_figure(stats_per_window, df_normalized, trace_name, x_values, x_label, figure_path, window_size, sliding_increment):
    avg_size = [] # to reorder the label
    # plot lines
    # The x-axis should use the original value, not the normalized one
    plt.plot(x_values, list(df_normalized['n_offset']),
        label = "#Unique addr (max = "+ str(stats_per_window['n_offset'].max()) +")", linestyle="dotted", linewidth=2)
    avg_size.append(((df_normalized['n_offset'].mean()), 0))
    plt.plot(x_values, list(df_normalized['throughput']), 
        label = "Thrpt (max = "+ str(stats_per_window['throughput'].max()) +" MBps)", linestyle="dashed", color="cyan")
    avg_size.append(((df_normalized['throughput'].mean()), 1))
    plt.plot(x_values, list(df_normalized['read_percent']), 
        label = "Percent of Read IO (%)", linestyle="--")
    avg_size.append(((df_normalized['read_percent'].mean()), 2))
    plt.plot(x_values, list(df_normalized['size_mb']), 
        label = "Size (max = "+ str(stats_per_window['size_mb'].max()) +" MB)", linestyle="-.")
    avg_size.append(((df_normalized['size_mb'].mean()), 3))
    plt.plot(x_values, list(df_normalized['iops']), 
        label = "IOPS (max = "+ str(stats_per_window['iops'].max()) +")", linestyle="dashed", linewidth=2)
    avg_size.append(((df_normalized['iops'].mean()), 4))
    plt.plot(x_values, list(df_normalized['latency']), 
        label = "Latency (max = "+ str(stats_per_window['latency'].max()) +" us)", linestyle="dashed", linewidth=1)
    avg_size.append(((df_normalized['latency'].mean()), 5))

    # plt.locator_params(axis='x', nbins=8)

    plt.ylim(0,1)
    plt.xlim(0,max(x_values))
    plt.xlabel(x_label)
    plt.ylabel("normalized_value")
    plt.title("Per-window characteristics of " + trace_name + "\n" + "[window_size = " + str(int(window_size)) + " IOs, increment = " + str(sliding_increment) + "]")
    plt.grid(axis='y')
    plt.grid(axis='x')
    fig = plt.gcf()
    fig.set_size_inches(15, 6)

    # Reorder legends 
    avg_size.sort(reverse=True)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [y for x,y in avg_size ]
    
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    plt.figure().clear() 

def plot_cdf(data, x_label, figure_path):
    N=len(data)
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)
    plt.xlabel(x_label)
    plt.ylabel('CDF')
    plt.title('CDF of ' + x_label)
    plt.plot(x_1, y_1, label = "raw (baseline)", color="red")
    plt.ylim(0,1)
    p70_lat = np.percentile(x_1, 70)
    plt.xlim(0, max(p70_lat * 3, 2000)) # Hopefully the x axis limit can catch the tail
    # plt.xlim(1, 2000)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(15, 6)
    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    plt.figure().clear() 

def analyze_profile(input_path):
    parent_dir = str(Path(input_path).parent)
    trace_name = os.path.basename(input_path) # with extension

    df = pd.read_csv(input_path, header=None, sep=',')  
    df.columns = ["ts_record","latency","io_type","size","offset","ts_submit","size_after_replay"]
    bad_rows = df[df['size'] != df['size_after_replay']]
    assert(len(bad_rows) == 0) # The replay process was bad, thus some IOs has different size than expected
    df = df.drop(columns=['ts_record', 'size_after_replay'])  # drop unnecessary columns

    n_window = 50
    # the higher the n_subwindow, the smoother the line
    n_subwindow = 10 # a window will be splitted into multiple sub window to speedup computation
    window_size = len(df)/n_window
    extension_str = "per_" + str(int(window_size)) + ""
    sliding_increment = int(window_size / n_subwindow) # the increment size of the window
    
    # build the DP
    stats_dp = [] # per sub window
    l, r = 0, sliding_increment
    while r <= len(df):
        curr_window = df.iloc[l:r]
        read_freq = curr_window['io_type'].value_counts().to_dict()
        tot_size = int(curr_window['size'].sum()/1000000) # MB
        duration = (curr_window.iloc[-1]["ts_submit"] - curr_window.iloc[0]["ts_submit"])/1000 # secs
        tot_latency = int(curr_window['latency'].sum()) # in us = microsecond
        timestamp = curr_window['ts_submit'].median() /1000 # secs
        start_time = curr_window.iloc[0]['ts_submit'] /1000 # secs
        end_time = curr_window.iloc[-1]['ts_submit'] /1000 # secs
        # [n_read, tot_size, duration, #unique offset, tot_latency, mid_timestamp, start_time, end_time]
        stats_dp.append([read_freq.get(1, 0), tot_size, duration, curr_window['offset'].nunique(), tot_latency, timestamp, start_time, end_time ])
        
        l += sliding_increment
        r += sliding_increment

    # Note: The #unique offset here is just approximation, and might not be accurate (window-wise)
    # Init value of the first window    
    n_read, tot_size, duration, n_offset, tot_latency = 0,0,0,0,0
    for i in range(0, n_subwindow):
        n_read += stats_dp[i][0]
        tot_size += stats_dp[i][1]
        duration += stats_dp[i][2]
        n_offset += stats_dp[i][3]
        tot_latency += stats_dp[i][4]

    # begin sliding the window
    stats_per_window = pd.DataFrame(columns=['window_id', 'start_idx', 'end_idx', 'read_percent', 'size_mb', 'throughput', 'iops', 'n_offset', 'latency', 'timestamp', 'start_time', 'end_time'])
    l, r = 0, n_subwindow - 1
    mid = n_subwindow//2
    while r < len(stats_dp):
        # print(l,r)
        # exit()
        # R ratio, total size, throughput, IOPS, #unique offset, latency, timestamp, start_time, end_time
        read_ratio = int((n_read/window_size)*100)
        start_time, end_time = stats_dp[l][6], stats_dp[r][7]
        stats_per_window.loc[l] = [l, l * sliding_increment, 
            r * sliding_increment, read_ratio, tot_size, round(tot_size/duration,2),
            round(window_size/(duration),2), # IOPS calculation per millisecond
            n_offset, 
            round(tot_latency/window_size,2), 
            stats_dp[l + mid][5], # timestamp in secs
            start_time, end_time] # in secs
        n_read += stats_dp[r][0] - stats_dp[r - n_subwindow][0]
        tot_size += stats_dp[r][1] - stats_dp[r - n_subwindow][1]
        duration += stats_dp[r][2] - stats_dp[r - n_subwindow][2]
        n_offset += stats_dp[r][3] - stats_dp[r - n_subwindow][3]
        tot_latency += stats_dp[r][4] - stats_dp[r - n_subwindow][4]  # in us = microsecond
        l += 1
        r += 1
    
    stats_per_window = stats_per_window.astype( dtype={'window_id' : int, 
                 'start_idx': int,
                 'end_idx': int,
                 'read_percent': int,
                 'size_mb': int,
                 'iops' : int,
                 'n_offset': int,
                 'timestamp': int, # in secs
                 'start_time' :int, # in secs
                 'end_time' : int}) # in secs
    stats_per_window["trace_name"] = str(trace_name)
    # stats_out_path = os.path.join(parent_dir, trace_name + ".csv")
    # write_to_file(stats_per_window, stats_out_path)

    # Normalize data for plotting
    df_normalized = stats_per_window.drop(columns=['trace_name']) # must be dropeed before normalization
    df_normalized=(df_normalized)/(df_normalized.max())

    # Create the per-window Figure
    filename = str(Path(trace_name).with_suffix('') ) # remove .trace extension

    figure_path1 = os.path.join(parent_dir, filename + ".window.png")
    plot_figure(stats_per_window, df_normalized, trace_name, list(df_normalized.index.values), "window_id", figure_path1, window_size, sliding_increment)

    figure_path2 = os.path.join(parent_dir, filename + ".time.png") 
    plot_figure(stats_per_window, df_normalized, trace_name, list(stats_per_window['timestamp']), "timestamp (s)", figure_path2, window_size, sliding_increment)

    # Create the Latency CDF
    figure_cdf_path = os.path.join(parent_dir, filename + ".cdf_lat.png")
    plot_cdf(list(df['latency']), 'Latency (us)', figure_cdf_path)

    # Combine all figures
    list_im = [figure_cdf_path, figure_path1, figure_path2]
    imgs    = [ Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.vstack([i.resize(min_shape) for i in imgs]) # for a vertical stacking it is simple: use vstack
    imgs_comb = Image.fromarray( imgs_comb)

    figure_path_final = os.path.join(parent_dir, filename + ".png") 
    imgs_comb.save( figure_path_final )
    print("===== output figure : " + figure_path_final)

    # Delete figures after we combine them into a single figure
    os.remove(figure_cdf_path)
    os.remove(figure_path1)
    os.remove(figure_path2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-files", help="List of file path of the trace", nargs='+',type=str)
    parser.add_argument("-file", help="File path of the trace",type=str)
    args = parser.parse_args()
    if (not args.file and not args.files):
        print("    ERROR: You must provide these arguments: -file <the input trace>")
        exit(-1)
    arr_profiles = []
    if args.files:
        arr_profiles += args.files
    elif args.file:
        arr_profiles.append(args.file)
        
    print("arr_profiles = " + str(arr_profiles))
    for input_path in arr_profiles:
        # print(input_path)
        analyze_profile(input_path)
