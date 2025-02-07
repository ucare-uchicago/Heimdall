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

sys.path.insert(0, "../script/")
import pattern_checker
import default_ip_finder

colors = {}
colors['baseline'] = 'red'
colors['base_trace_1'] = 'orangered'
colors['base_trace_2'] = 'darkred'
colors['ideal'] = 'grey'
colors['c3'] = "orange"
colors['linnos'] = "blue"
colors['linnos_hedging'] = "blue"
colors['flashnet'] = "green"
colors['flashnet_hedging'] = "green"
colors['random'] = "purple"
colors['flashnet_rw_hist'] = "magenta"

linestyles = {}
linestyles['ideal'] = 'dashed'
linestyles['base_trace_1'] = 'dotted'
linestyles['base_trace_2'] = 'dotted'
linestyles['linnos_hedging'] = 'dotted'
linestyles['flashnet_hedging'] = 'dotted'
linestyles['flashnet_rw_hist'] = "dashed"

linewidths = {}
linewidths['ideal'] = 1
linewidths['base_trace_1'] = 0.5
linewidths['base_trace_2'] = 0.5
linewidths['flashnet_rw_hist'] = 1.5

FONT_SIZE = 6

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

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    # print("===== output file : " + filePath)

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

def plot_cdf(data, algo_name, x_label, title, figure_path):
    N=len(data)
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)
    plt.xlabel(x_label)
    plt.ylabel('CDF')

    # set smaller font size for title
    plt.title(title, fontsize=FONT_SIZE)

    plt.plot(x_1, y_1, label = algo_name, color=colors.get(algo_name, 'black'), linestyle=linestyles.get(algo_name, 'solid'), linewidth=linewidths.get(algo_name, 1))

    plt.ylim(0,1)
    p50_lat = np.percentile(x_1, 70)
    plt.xlim(0, max(p50_lat * 3, 500)) # Hopefully the x axis limit can catch the tail
    # plt.xlim(1, 2000)
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(4, 3)

    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    plt.close()
    plt.figure().clear() 

def get_algo_dirs(trace_dir):
    # list all dirs in this path
    algo_dirs = []
    algo_names = []
    for path in os.listdir(trace_dir):
        path = os.path.join(trace_dir, path)
        if os.path.isdir(path):
            algo_dirs.append(path)
            algo_names.append(os.path.basename(path))
    return algo_dirs, algo_names

def get_all_trace_path(trace_dir):
    traces_in_this_dir = []
    for trace in os.listdir(trace_dir):
        trace_path = os.path.join(trace_dir, trace)
        if os.path.isfile(trace_path) and str(trace).endswith('.trace'):
            traces_in_this_dir.append(trace_path)
    return traces_in_this_dir

def split_long_path_to_multiple_lines(algo_dir):
    parent_dirs = ""
    # print(path)
    # get the relative path
    trace_dir = os.path.abspath(algo_dir)
    # print("trace_dir = " + trace_dir)
    # generate the title
    if "data/" in trace_dir:
        # parent_dirs = str(Path(trace_dir).parent).split('/')
        parent_dirs = str(trace_dir).split('/')
        # find parent dir that contains "grouping"
        for i in range(len(parent_dirs)):
            if "data" in parent_dirs[i]:
                break
        parent_dirs = parent_dirs[i:]
        # split long title in three lines
        last_line_str = parent_dirs[4:]
        if len(last_line_str) > 3:
            # split it into two lines
            last_line_str = last_line_str[:2] + ["\n"] + last_line_str[2:]
        parent_dirs_str =  "/".join(parent_dirs[0:2]) + "\n" + "/".join(parent_dirs[2:4]) + "\n" + "/".join(last_line_str)
        return parent_dirs_str
    else:
        print("ERROR: The trace_dir must be in data/")
        exit(-1)

def write_stats_per_trace(aggregated_latencies, output_path):
    stats = []
    # get p1,p2, ..., p100 of the latency
    percentiles = [x for x in range(1, 101)]
    lat_percentiles = np.percentile(aggregated_latencies, percentiles)
    lat_percentiles = np.around(lat_percentiles, decimals=2)
    stats.append("percentile\tlatency_us")
    for idx, p in enumerate(percentiles):
        stats.append(str(p) + "\t" + str(lat_percentiles[idx]))

    # write stats to a file
    write_stats(output_path, "\n".join(stats))
    return lat_percentiles

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
        arr_read_latency.append(list(df['latency']))
        arr_trace_names.append(filename)
    return arr_read_latency, arr_trace_names

# Only draw CDF for read IOs!!!
def draw_cdf_per_algo(algo_dir):
    algo = os.path.basename(algo_dir)
    
    traces_in_this_dir = [trace for trace in os.listdir(algo_dir) if (os.path.isfile(os.path.join(algo_dir, trace)) and str(trace).endswith('.trace'))]

    parent_dirs = split_long_path_to_multiple_lines(algo_dir)
    aggregated_latencies = [] # aggregated read latencies of all traces in this dir

    for trace_name in traces_in_this_dir:
        input_path = os.path.join(algo_dir, trace_name)
        filename = str(Path(trace_name).with_suffix('') ) # remove .trace extension
        figure_path =  os.path.join(algo_dir, "cdf_" + filename + ".png")

        # check if we have access to this dir
        if not os.access(algo_dir, os.W_OK):
            print("   WARNING: I don't have access to this dir ")
            print("       The data is currently being replayed in here.")
            print("       Skipping this algo dir")
            continue

        df = read_replayed_file(input_path)

        # draw the trace charac

        # Only draw CDF for read IOs!!!
        df = df[df['io_type'] == 1]

        if len(df) == 0:
            print("\nERROR: The trace file " + trace_name + " doesn't have any read IOs")
            print("algo_dir = " + algo_dir)
            print("\tDelete this trace combination and re-run the script!")
            exit(-1)

        bad_rows = df[df['size'] != df['size_after_replay']]
        if len(bad_rows) > 0:
            print("WARNING: The replay process was bad, thus some IOs has different size than expected")
            print("There are " + str(len(bad_rows)) + " bad rows")
            # print(bad_rows)
        assert(len(bad_rows) == 0) # The replay process was bad, thus some IOs has different size than expected
        df = df.drop(columns=['ts_record', 'size_after_replay'])  # drop unnecessary columns

        title =  filename + ' CDF of ' + parent_dirs
        plot_cdf(df['latency'], algo, 'Latency (us)', title, figure_path)
        aggregated_latencies += list(df['latency'])

    if aggregated_latencies == []:
        print("Skipping this dir ")
        return None, None

    # write stats for aggregated latency
    output_path = os.path.join(algo_dir, "latency_aggregated.stats")
    lat_percentiles = write_stats_per_trace(aggregated_latencies, output_path)

    # Create the aggregated CDF
    title =  'Aggregate CDF of ' + parent_dirs
    figure_path =  os.path.join(algo_dir, "cdf_aggregated" + ".png")
    plot_cdf(aggregated_latencies, algo, 'Latency (us)', title, figure_path)

    return aggregated_latencies, lat_percentiles

def plot_multi_line_cdf(per_algo_latencies, title, figure_path):
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    # fig.set_size_inches(4, 3)
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    # anchor the title slightly to the right
    plt.title(title, fontsize=FONT_SIZE, x=0.75)

    x_lim = 0
    x_lim_baseline = None
    for algo_name in per_algo_latencies:
        data = per_algo_latencies[algo_name]
        N=len(data)
        # sort the data in ascending order
        x_1 = np.sort(data)
        # get the cdf values of y
        y_1 = np.arange(N) / float(N)
        # set smaller font size for title
        plt.plot(x_1, y_1, label = algo_name, color=colors.get(algo_name, 'black'), linestyle=linestyles.get(algo_name, 'solid'), linewidth=linewidths.get(algo_name, 1))
        plt.ylim(0,1)

        if algo_name == 'baseline':
            # 20x median
            x_lim_baseline = np.percentile(x_1, 50) * 20
        else:
            p50_lat = np.percentile(x_1, 50)
            x_lim = max(p50_lat * 20, x_lim)

    if x_lim_baseline != None:
        # 20x median of the baseline
        plt.xlim(0, x_lim_baseline)
    else:
        plt.xlim(0, x_lim) 

    plt.ylim(0,1)
    # plt.legend(loc="lower right")
    # put the legend outside the plot in the right side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    print("===== output figure : " + figure_path)
    plt.close()
    plt.figure().clear() 


def plot_multi_line_cdf_clean(per_algo_latencies, title, figure_path):
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    # fig.set_size_inches(4, 3)
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    # anchor the title slightly to the right
    plt.title(title, fontsize=FONT_SIZE, x=0.75)

    x_lim = 0
    x_lim_baseline = None
    for algo_name in per_algo_latencies:
        if "base_" in algo_name:
            continue
        data = per_algo_latencies[algo_name]
        N=len(data)
        # sort the data in ascending order
        x_1 = np.sort(data)
        # get the cdf values of y
        y_1 = np.arange(N) / float(N)
        # set smaller font size for title
        plt.plot(x_1, y_1, label = algo_name, color=colors.get(algo_name, 'black'), linestyle=linestyles.get(algo_name, 'solid'), linewidth=linewidths.get(algo_name, 1))
        plt.ylim(0,1)

        if algo_name == 'baseline':
            # 5x median
            x_lim_baseline = np.percentile(x_1, 50) * 5
        else:
            p50_lat = np.percentile(x_1, 50)
            x_lim = max(p50_lat * 5, x_lim)

    if x_lim_baseline != None:
        # 5x median of the baseline
        plt.xlim(0, x_lim_baseline)
    else:
        plt.xlim(0, x_lim) 

    plt.ylim(0,1)
    # plt.legend(loc="lower right")
    # put the legend outside the plot in the right side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    plt.close()
    plt.figure().clear() 


def generate_cdf_title(trace_dir):
    parent_dirs = split_long_path_to_multiple_lines(trace_dir)
    title =  'Aggregate CDF of ' + parent_dirs
    return title

def cacl_latency_improvement(per_algo_100_percentiles):
    stats = []
    p_start = 70
    p_end = 95
    # get the baseline percentile
    baseline_percentiles = per_algo_100_percentiles['baseline']
    for algo_name in per_algo_100_percentiles:
        if algo_name == 'baseline':
            continue
        # get the percentile of this algo
        algo_percentiles = per_algo_100_percentiles[algo_name]
        # sum the difference of each percentile
        improvement = 0
        for percentile in range(p_start, p_end + 1): 
            improvement += (baseline_percentiles[percentile - 1] - algo_percentiles[percentile - 1])
        # average the improvement
        improvement = round(improvement / len(algo_percentiles), 2)
        stats.append( "(p" + str(p_start) + "-p" + str(p_end) + ") improvement algo " + algo_name + " over baseline \t= " + str(improvement))
    return stats

def calc_x_max_from_baseline(per_algo_latencies):
    if 'baseline' in per_algo_latencies:
        baseline_latencies = per_algo_latencies['baseline']
        x_max = np.median(baseline_latencies) * 5
        # print("baseline median = " + str(np.median(baseline_latencies)))
    else:
        x_max = None
    return x_max

def get_100_lat_percentiles(lat_array):
    percentiles = [x for x in range(1, 101, 1)]
    n = len(percentiles)
    # print("percentiles = " + str(percentiles))
    lat_percentiles = np.percentile(lat_array, percentiles)
    lat_percentiles = np.around(lat_percentiles, decimals=2)
    return lat_percentiles

def get_inflection_points_stats(per_algo_latencies):
    stats_percent = []
    stats_latency = []
    stats_alignment = []
    stats_percent.append("=====================================")
    stats_latency.append("=====================================")
    stats_alignment.append("=====================================")

    x_max = calc_x_max_from_baseline(per_algo_latencies)
    baseline_left_area_fast_section = None
    baseline_ip_percent = None

    # get the inflection point of each algo
    for algo_name in per_algo_latencies:
        latencies = per_algo_latencies[algo_name]
        ip_latency_threshold, ip_latency_percent = default_ip_finder.area_based(latencies, x_max)
        stats_percent.append("IP percent algo " + algo_name + " \t= " + str(round(ip_latency_percent, 2)) + " %")
        stats_latency.append("IP latency algo " + algo_name + " \t= " + str(int(ip_latency_threshold)) + " us")
        if algo_name == 'baseline':
            baseline_ip_percent = ip_latency_percent
            baseline_left_area_fast_section = default_ip_finder.calc_area_going_left(latencies, ip_latency_percent)
            stats_alignment.append("Baseline fast section area size (going left) \t= " + str(round(baseline_left_area_fast_section,2)))
            fast_lat_mid_point = round(baseline_left_area_fast_section /baseline_ip_percent , 2)
            stats_alignment.append("Baseline fast latency mid point\t= " + str(fast_lat_mid_point) + " us")

            # Get the characteristics of the fast section of the baseline
            lat_100_percentiles = get_100_lat_percentiles(latencies)
            fast_latencies = [lat for lat in lat_100_percentiles if lat <= ip_latency_threshold]
            fast_lat_var = round(np.var(fast_latencies), 2)
            stats_alignment.append("Baseline fast latency variance \t= " + str(fast_lat_var) + " us")
            fast_lat_std = round(np.std(fast_latencies), 2)
            stats_alignment.append("Baseline fast latency std      \t= " + str(fast_lat_std) + " us")

            # count how many latencies within 1 std of the mid point
            fast_lat_within_1_std = [lat for lat in fast_latencies if (lat >= fast_lat_mid_point - fast_lat_std and lat <= fast_lat_mid_point + fast_lat_std)]
            # print("fast_latencies = " + str(fast_latencies))
            # print(fast_lat_within_1_std)

            fast_lat_stability_v0 = round(len(fast_lat_within_1_std) / len (fast_latencies) * 100, 2)

            # get the std percent of the mid point
            fast_lat_stability_v1 = round(100 - (fast_lat_std / fast_lat_mid_point * 100), 2)
            stats_alignment.append("Baseline fast latency stability\t= " + str(fast_lat_stability_v1) + " %")
            # print("baseline_left_area_fast_section = " + str(baseline_left_area_fast_section))
    
    # measure the alignment of the lines under the baseline's IP 
    if baseline_left_area_fast_section != None:
        for algo_name in per_algo_latencies:
            if algo_name == 'baseline':
                continue
            # print(" baseline_ip_percent = " + str(baseline_ip_percent))
            latencies = per_algo_latencies[algo_name]
            left_area_fast_section = default_ip_finder.calc_area_going_left(latencies, baseline_ip_percent)
            # print(" left_area_fast_section = " + str(left_area_fast_section))
            alignment_diff = (baseline_left_area_fast_section - left_area_fast_section) / baseline_left_area_fast_section * 100
            # print("alignment_diff of " + algo_name + " " + str(alignment_diff))
            stats_alignment.append("Alignment diff " + algo_name + " (vs baseline) \t= " + str(round(alignment_diff, 2)) + " %")
    return stats_percent + stats_latency + stats_alignment

def is_all_algo_analyzed(arr_algo_dirs):
    # check if all algo dirs have the CDF generated
    for algo_dir in arr_algo_dirs:
        if not os.path.exists(os.path.join(algo_dir, "cdf_aggregated.png")):
            return False

    # Check if the all_algo_eval.stats mentions all the algos
    parent_dir = os.path.dirname(arr_algo_dirs[0])
    stats_path = os.path.join(parent_dir, "all_algo_eval.stats")
    if not os.path.exists(stats_path):
        return False
    with open(stats_path, 'r') as file:
        data = file.read()
        for algo_dir in arr_algo_dirs:
            algo_name = os.path.basename(algo_dir)
            if algo_name not in data:
                return False
    return True

def is_cdf_outdated(arr_algo_dirs):
    # check if it has the cdf_all_algo.png
    parent_dir = os.path.dirname(arr_algo_dirs[0])
    cdf_all_algo_path = os.path.join(parent_dir, "cdf_all_algo.png")
    if not os.path.exists(cdf_all_algo_path):
        # this is a new trace combination, we need to generate the CDF
        return True
    else:
        # check the modified time of the cdf_all_algo.png
        cdf_all_algo_modified_time = os.path.getmtime(cdf_all_algo_path)
        for algo_dir in arr_algo_dirs:
            # check the modified time of the trace stats 
            trace_stats_path_2 = os.path.join(algo_dir, "trace_2.trace.stats")
            trace_stats_path_1 = os.path.join(algo_dir, "trace_1.trace.stats")
            if os.path.getmtime(trace_stats_path_2) > cdf_all_algo_modified_time or os.path.getmtime(trace_stats_path_1) > cdf_all_algo_modified_time:
                # this algo is just being replayed, the current CDF is outdated
                return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-trace_dir", help="File path to the trace sections description ", type=str)
    parser.add_argument("-trace_dirs", help="Arr of file path to the trace sections description", nargs='+',type=str)
    parser.add_argument("-reverse", help="Reverse the order of the traces", action='store_true')
    parser.add_argument("-resume", help="Only run the analysis if there is newly run algorithm", action='store_true')
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
                if is_cdf_outdated(arr_algo_dirs):
                    print("===== One of the CDF is outdated; re-generating CDF in this trace combination")
                else:
                    print("===== Existing CDF = " + str(trace_dir) + "/cdf_all_algo.png")
                    continue

        for idx, algo_dir in enumerate(arr_algo_dirs):
            algo_name = arr_algo_names[idx]
            # draw per algo CDFs
            arr_read_latency, lat_percentiles = draw_cdf_per_algo(algo_dir)
            if arr_read_latency == None:
                # we don't have access to this dir; replaying is still in progress
                continue
            per_algo_latencies[arr_algo_names[idx]] = arr_read_latency
            per_algo_100_percentiles[arr_algo_names[idx]] = lat_percentiles
            
            if algo_name == 'baseline':
                # get the read latencies from each traces in baseline 
                arr_read_latency, arr_trace_names = get_per_trace_latency(algo_dir)
                for idx, trace_name in enumerate(arr_trace_names):
                    # adding to the dict for the CDF plotting
                    per_algo_latencies["base_" + trace_name] = arr_read_latency[idx]
        
        if per_algo_100_percentiles == {}:
             # we don't have access to this dir; replaying is still in progress
            continue

        # get latency improvement of each algo compared to the baseline
        if 'baseline' in arr_algo_names:
            stats += cacl_latency_improvement(per_algo_100_percentiles)

        # Gather stats of the Inflection points 
        stats += get_inflection_points_stats(per_algo_latencies)

        # write the stats to a file
        output_path = os.path.join(trace_dir, "all_algo_eval.stats")
        write_stats(output_path, "\n".join(stats))

        # Create the aggregated CDF
        title = generate_cdf_title(trace_dir)
        # create CDF with multple lines 
        figure_path =  os.path.join(trace_dir, "cdf_all_algo" + ".png")
        plot_multi_line_cdf(per_algo_latencies, title, figure_path)
        figure_path =  os.path.join(trace_dir, "cdf_all_algo" + "_clean.png")
        plot_multi_line_cdf_clean(per_algo_latencies, title, figure_path)

