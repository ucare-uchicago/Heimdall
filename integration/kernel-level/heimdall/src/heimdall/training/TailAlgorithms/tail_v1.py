#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import bisect
from sklearn.metrics import auc

import default_ip_finder

# These all numbers should be the same [5, 4]
N_HISTORY = 3
N_FUTURE = 3

# Filtering slow IO
# MAX_IO_LATENCY = 300    # -1 for ignoring this cut-off threshold
# THPT_IP_CORRECTION = 2
# LAT_INCREASE_RATE = 1.5
THPT_DROP_RATE = 1.7

# MAX_THPT_INCREASE = 2
# MIN_SIZE_FOR_GC_START = 20000
# MIN_SIZE_FOR_GC_START = 5000

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(df, filePath, has_header=True):
    # The raw (replayed) traces don't have the header
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def read_file(input_file):
    df = pd.read_csv(input_file, header=None, sep=',')
    # Make sure it has 7 columns
    assert 7 == df.shape[1]
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B), io_ts(ms)
    df.columns = ["ts_record","latency","io_type","size","offset","ts_submit","size_after_replay","io_ts"]

    # filter: remove io that doesn't executed properly (can't read/write all bytes)
    df = df[df['size'] == df['size_after_replay']]
    return df

#  This will turn a plain latency dataframe into a collection of CDF latency per IO-size
def build_cdf_dict(latency_df):
    lat_cdf = {}
    for io_size in latency_df['size'].unique():
        list_of_lat = list(latency_df[latency_df['size'] == io_size]['latency'].values)
        list_of_lat.sort()
        # now, map the idx position to the new dict
        lat_to_idx_map = {}
        n_total = len(list_of_lat)
        for idx, lat in enumerate(list_of_lat):
            if lat not in lat_to_idx_map:
                # add to the mapping
                lat_to_idx_map[lat] = round(idx/n_total, 2)

        # save that lat to idx mapping to the CDF dict
        # print(set(list_of_lat))
        lat_cdf[io_size] = lat_to_idx_map
    return lat_cdf

# FEATURE 1: add N last latency
def collect_history(df, col_name, n_history = N_HISTORY):
    # print(len(df))
    history_holder = pd.DataFrame()
    for n in range(1, n_history + 1):
        # get the history (adding 0 for the first IOs that doesn't have)
        values = ([0] * n) + list(df[col_name].values)
        values = values [:len(values) - n] # remove extra value
        history_holder[n] = values
    history_holder['all_history']= history_holder.values.tolist()
    # print(history_holder)
    # print(len(history_holder['all_history']))
    # exit()
    return history_holder['all_history'].values


# FEATURE 1: add N future entry (The current IO is regarded as part of the future)
def collect_future(df, col_name, n_future = N_FUTURE):
    # print(len(df))
    future_holder = pd.DataFrame()
    for n in range(0, n_future):
        # get the future (adding 0 for the last IOs that doesn't have)
        values = list(df[col_name].values) + ([0] * n) 
        values = values [n:len(values)] # remove extra value
        future_holder[n] = values
    future_holder['all_future']= future_holder.values.tolist()
    return future_holder['all_future'].values

def mark_possible_start_1(row, ip_latency_threshold, ip_throughput_threshold, thpt_drop_rate):
    if (row['throughput'] > ip_throughput_threshold or row['latency'] < ip_latency_threshold):
        # This IO is definitely fast enough, thus, can't be the GCstart
        return " "
    else:
        if ((row['throughput_drop'] >= thpt_drop_rate)):
            return " GC-Start1 " 
        else:
            return " "
      
def merge_consecutive_gc(df):
    # Merge back-to-back GC period
    max_idx = len(df) - 1
    idx = 1
    # Iterate the dataframe
    while(idx <= max_idx):
        row = df.iloc[idx]
        prev_row = df.iloc[idx-1]
        # Will start processing at " GC-Start " marker
        if prev_row['mark_gc_end'] == " GC-End " and row["mark_gc_start"] == " GC-Start2 ":
            # remove both marks
            df.at[idx, "mark_gc_start"] = "  "
            df.at[idx-1, "mark_gc_end"] = "  "
        idx += 1
    return df

def plot_raw_vs_best(figure_path, y_raw, y_best, extra_info=""):
    # Draw CDF
    N=len(y_best)
    data = y_best
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    N=len(y_raw)
    data = y_raw
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)

    # plotting
    plt.figure(figsize=(7,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title('CDF of Latency (Read-only IOs) \n' + extra_info)
    p70_lat = np.percentile(y_raw, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw latency", color="red")
    plt.plot(x_1, y_1, label = "FlashNet-best-case", linestyle='dashdot', color="green")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    print("===== output figure : " + figure_path)

def calc_percent(partition, total, precision = 2):
    return str(round(partition*100/total,precision)) + "%"

def calc_cdf_gain(y_raw, y_best):
# 1. The Raw Value
    # sort the data in ascending order
    N=len(y_raw)
    x_2 = np.sort(y_raw)
    y_2 = np.arange(N) / float(N)

    # Must limit the x axis, we don't want to calculate the area of the insignificant tail
    p70_lat = np.percentile(y_raw, 70)
    x_limiter = max(p70_lat * 3, 1000) # same as how we draw the CDF above
    idx = bisect.bisect_left(x_2, x_limiter)
    x_2 = x_2[:idx]
    y_2 = y_2[:idx]
    max_tail_value = x_2[-1]

# 2. The BEST-case Value
    # sort the data in ascending order
    N=len(y_best)
    x_1 = np.sort(y_best)
    y_1 = np.arange(N) / float(N)

    # Must limit the x axis
    idx = bisect.bisect_left(x_1, x_limiter)
    print(idx)
    x_1 = x_1[:idx]
    y_1 = y_1[:idx]

    # Must add padding to make sure that it reach the x limit
    if max(x_1) < max_tail_value:
        x_1 = np.append(x_1, max_tail_value) 
        y_1 = np.append(y_1, 1) 
    
    # print(len(y_raw), len(x_2))
    # print(len(y_best), len(x_1), len(y_1))
    
# 3. Calculate the AUC
    cdf_raw = auc(x_2, y_2)
    cdf_best = auc(x_1, y_1)

    # print('Raw CDF area  : {}'.format(cdf_raw))
    # print('Best CDF area : {}'.format(cdf_best))
    percent_gain = calc_percent(cdf_best - cdf_raw, cdf_raw)
    # print(percent_gain)

    # plt.figure(figsize=(7,3))
    # plt.xlabel('Latency (us)')
    # plt.ylabel('CDF')
    # plt.xlim(0, x_limiter) # Hopefully the x axis limit can catch the tail
    # # plt.ylim(0, 1) 
    # plt.plot(x_2, y_2, label = "Raw latency", color="red")
    # plt.plot(x_1, y_1, label = "FlashNet-best-case", color="green")
    # plt.legend(loc="lower right")
    # plt.savefig("figure_path.png", bbox_inches='tight')
    # print("===== output figure : " + "figure_path.png")
    # print(percent_gain)
    # exit(0)
    return percent_gain
    # Note: Do not use np.trapz(xx,yy), it doesn't calculate valid area
    
def start_processing(input_path,output,device): 
# 1. Add more variable to Analyze the Trace
    df = pd.read_csv(input_path)
    df.columns=['ts_record','latency','io_type','size','offset','ts_submit','device','io_ts']
    df = df[df['device'] == int(device)] 
    df = df.drop(columns=["device"], axis=1)
    stats_total_io = len(df)
    stats_n_read = len(df[df["io_type"] == 1])

    # Sort based on ts_submit, there is a slight out of order due to multithreading submission
    df = df.sort_values('ts_submit')
    df = df.reset_index(drop=True)

    # put a separator
    df["sep"] = "  "
    
    # add throughput
    df['throughput'] = df['size']/df['latency']
    df['throughput'] = df['throughput'].round(0)

    # put a separator
    df["sep2"] = "  "

# 2. Find SLOW throughput and Get the Latency Threshold 
    ip_latency_threshold, ip_latency_percent = default_ip_finder.tangent_based(df['latency'])
    # if throughput is higher, it is definitely FAST IO
    ip_throughput_threshold, ip_thpt_percent = default_ip_finder.tangent_based(df['throughput'])

    if (ip_latency_percent < 50):
        print("ERROR: this trace profile is BAD because the IP latency is < 50%. Flashnet won't be able to make any significant improvement.")

    # slow throughput if it is less than the median throughput
    median_throughput = np.percentile(df['throughput'], 50)
    # if less than median_latency, the IO is FAST
    median_latency = np.percentile(df['latency'], 50)
    print("IP latency threshold : " + str(ip_latency_threshold)+ " (" + str(ip_latency_percent)+ "%)")
    print("Median throughput threshold : " + str(median_throughput) + " (" + str(50)+ "%)")
    print("IP throughput threshold : " + str(ip_throughput_threshold) + " (" + str(ip_thpt_percent)+ "%)")
    print("Median latency threshold : " + str(median_latency) + " (" + str(50)+ "%)")
    # correction = 0.8 Means that, we reduce the IP percentile by 0.2 or 20%
    # slow_throughput_threshold = np.percentile(df['throughput'], ip_thpt_percent) # * THPT_IP_CORRECTION
    # print ("   true percentile = " + str(ip_thpt_percent) + "; corrected slow_throughput_threshold = " + str(slow_throughput_threshold))

# 3. Find the GC-Start 
    df["n_hist_throughput"] = collect_history(df, "throughput", n_history=N_HISTORY)
    df["n_hist_latency"] = collect_history(df, "latency", n_history=N_HISTORY)
    
    # Check based on the current vs previous latency
    # df["latency_increase"] = df.apply (lambda row: round(row['latency'] / (row['n_hist_latency'][0] + 1), 1) , axis=1)

    # Check based on the current vs previous throughput
    df["throughput_drop"] = df.apply (lambda row: round(row['n_hist_throughput'][0] / (row['throughput'] + 0.1), 1) , axis=1)

    # DAN: IMPORTANT VARIABLE for tuning
    # lat_increase_rate = LAT_INCREASE_RATE   # lower than this, it is not in the GC region
    thpt_drop_rate = THPT_DROP_RATE      # lower than this, it is not in the GC region
    # analyze the latency_increase and throughput_drop; 
        # will also use ip_latency; the gc-start should be higher than the ip_latency
    df["mark_start1"] = df.apply (lambda row: mark_possible_start_1(row, ip_latency_threshold, ip_throughput_threshold, thpt_drop_rate), axis = 1)

    # # Check based on next throughput, the next should be 2x larger!
    df["n_future_throughput"] = collect_future(df, "throughput", n_future=N_FUTURE)
    # df["mark_gc_start"] = df.apply (lambda row: find_gc_start(row, max_thpt_increase), axis = 1)

# 4. Find the GC-END
    # GC-END = If N_FUTURE (3) consecutive IOs has throughput higher than median
    n_slow_io = 0
    # Iterate starting at GC-Start1
    df["mark_gc_end"] = df.apply (lambda row: "  ", axis=1)
    max_idx = len(df) - 1
    idx = 0
    # Iterate the dataframe
    while(idx <= max_idx):
        row = df.iloc[idx]
        # Will start processing at " GC-Start " marker
        if row["mark_start1"] == " GC-Start1 ":
            n_slow_io += 1
            df.at[idx, "mark_tail"] = " Tail-Period " # Mark the START
            # going down checking the future thpt
            while(idx < max_idx):
                idx += 1
                row = df.iloc[idx]
                if all(i >= median_throughput for i in row['n_future_throughput']):
                    # if Yes, it is the GC-END; no need to mark anything
                    break 
                else:
                    n_slow_io += 1
                    df.at[idx, "mark_tail"] = " Tail-Period " # Mark it as slow
        # check next row until finding the starting point of the GC
        idx += 1
    print("n_slow_io = " + str(n_slow_io))

# 5. Mark outlier in between GC period
    # 5.1 Outlier = Latency under the median_latency

    df["mark_outlier"] = df.apply (lambda row: "  ", axis=1)
    
    max_idx = len(df) - 1
    n_outlier1 = 0
    n_outlier2 = 0
    idx = 0
    # Iterate the dataframe to mark the outlier
    while(idx <= max_idx):
        row = df.iloc[idx]
        if row["mark_tail"] == " Tail-Period ":
            # SLOW IO category
            if row['latency'] <= median_latency and row['throughput'] >= median_throughput:
                # Fast IO should NOT be within the tail period
                df.at[idx, "mark_outlier"] = " outlier1 "
                n_outlier1 += 1
        else:
            # FAST IO category
            # Very slow IO should NOT be here
            if row['latency'] > ip_latency_threshold:
                # Check the throughput, maybe it is heavy (io_size is big)
                if row['throughput'] < median_throughput:
                    df.at[idx, "mark_outlier"] = " outlier2 "
                    n_outlier2 += 1
        idx += 1

    print("Outlier within slow period = " + str(n_outlier1))
    print("Outlier within fast period = " + str(n_outlier2))

    # Remove outlier 
    df = df[df['mark_outlier'] == "  "]
    df = df.reset_index(drop=True)

# 6. Remove Outlier spike
    # Must be done after removing outlier1 and outlier2
    # Remove tail that only has IO <= the N_HISTORY
    max_idx = len(df) - 1
    n_outlier3 = 0
    idx = 0
    # Iterate the dataframe
    while(idx <= max_idx):
        row = df.iloc[idx]
        # Will start processing at " GC-Start " marker
        if row["mark_tail"] == " Tail-Period ":
            n_tail = 1 
            # going down checking the next slow IOs
            while(idx < max_idx):
                idx += 1
                row = df.iloc[idx]
                if row["mark_tail"] != " Tail-Period ":
                    if n_tail <= N_HISTORY:
                        # mark this period as outlier
                        n_outlier3 += n_tail
                        while n_tail > 0:
                            df.at[idx - n_tail, "mark_outlier"] = " outlier3 "
                            n_tail -= 1
                    break 
                n_tail += 1
        idx += 1
    print("Outlier short tail spike = " + str(n_outlier3))

    # Remove outlier 
    df = df[df['mark_outlier'] == "  "]
    df = df.reset_index(drop=True)

# # 7. Create Ouput dir
#     # must include the rerated_*/ or resized_*/
#     list_dirs = input_path.split("/")
#     profile_name = os.path.basename(list_dirs.pop())
#     parent_dir =  []
#     for i, dirname in enumerate(list_dirs):
#         if dirname == "trace_profile":
#             parent_dir = list_dirs[(i+1):]

#     parent_dir = "/".join(parent_dir)
#     profile_name = str(Path(profile_name).with_suffix('') ) # remove .trace extension
#     output_dir = "../dataset/" + parent_dir + "/" + profile_name
#     create_output_dir(output_dir)

# 7. Write the marked data
    df = df.drop('n_hist_throughput', axis=1)
    df = df.drop('n_hist_latency', axis=1)
    df = df.drop('n_future_throughput', axis=1)
    stats_n_labeled = len(df)
    print("#IO labeled = " + str(stats_n_labeled))

    # outfile_path = os.path.join(output_dir, "profile_v1.marked")

    # write_to_file(df, outfile_path, True)
    # write_to_file(latency_df, outfile_path + ".tmp", True)

# 8. Write data as labeled dataset
    # drop unnecessary columns 
    important_columns = ["ts_record","latency","io_type","size","offset","ts_submit","size_after_replay", "mark_tail"]
    df = df.loc[:, df.columns.intersection(important_columns)]
    df["reject"] = df.apply (lambda row: 1 if(row['mark_tail'] == " Tail-Period ") else 0, axis=1)
    
    # drop marker column
    df = df.drop('mark_tail', axis=1)

    # outfile_path = os.path.join(output_dir, "profile_v1.labeled")
    stats_n_fast_io = len(df[df['reject'] == 0])
    stats_n_slow_io = len(df[df['reject'] == 1])
    print("Fast IO = " + str(stats_n_fast_io))
    print("Slow IO = " + str(stats_n_slow_io))
    # write_to_file(df, outfile_path, True)
    write_to_file(df, output, True)

# # 9. Filter out the Write IOs
#     df = df[df["io_type"] == 1] # Only check the read
#     stats_n_read_io_labeled = len(df)
#     stats_n_fast_read_io = len(df[df['reject'] == 0])
#     stats_n_slow_read_io = len(df[df['reject'] == 1])

# # 10. Calculate the CDF gain on Read-only IO
#     y_best = df.loc[df['reject'] == 0, 'latency']
#     y_raw = df['latency'].values
#     stats_cdf_gain = calc_cdf_gain( y_raw, y_best)

# # 11. Write the stats
#     logging = []
#     logging.append("============================================")
#     logging.append("                BASIC INFO ")
#     logging.append("============================================")
#     logging.append("Profile name = " + profile_name)
#     logging.append("Full path    = " + input_path)
#     stats_read_ratio = int((stats_n_read/stats_total_io)*100)
#     logging.append("R:W ratio    = " + str(stats_read_ratio) + ":" + str(100-stats_read_ratio))
#     logging.append("#IO          = " + str(stats_total_io))
#     logging.append("#writes      = " + str(stats_total_io - stats_n_read))
#     logging.append("#reads       = " + str(stats_n_read))
#     logging.append("============================================")
#     logging.append("                STATISTICS")
#     logging.append("============================================")
#     logging.append("IP latency        = " + str(ip_latency_threshold) + " us ("+ str(round(ip_latency_percent, 2)) +"%)")
#     logging.append("IP throughput     = " + str(ip_throughput_threshold) + " B/us ("+ str(round(ip_thpt_percent, 2)) +"%)")
#     logging.append("Median latency    = " + str(ip_latency_threshold) + " us (50%)")
#     logging.append("Median throughput = " + str(ip_latency_threshold) + " B/us (50%)")
#     logging.append("Outlier within slow period = " + str(n_outlier1) + "  ("+calc_percent(n_outlier1,stats_total_io)+")")
#     logging.append("Outlier within fast period = " + str(n_outlier2) + "  ("+calc_percent(n_outlier2,stats_total_io)+")")
#     logging.append("Outlier short tail spike   = " + str(n_outlier3) + "  ("+calc_percent(n_outlier3,stats_total_io)+")")
#     stats_n_outlier = n_outlier3 + n_outlier2 + n_outlier1
#     stats_percent_outlier = calc_percent(stats_n_outlier,stats_total_io)
#     logging.append("#Outlier IO  = " + str(stats_n_outlier) + "  ("+ stats_percent_outlier +" out of " + str(stats_total_io) + ")")
#     logging.append("#IO labeled  = " + str(stats_n_labeled) + "  ("+calc_percent(stats_n_labeled,stats_total_io)+" out of " + str(stats_total_io) + ")")

#     logging.append("  #Write IO  = " + str(stats_n_labeled - stats_n_read_io_labeled))
#     logging.append("  #Read IO   = " + str(stats_n_read_io_labeled))

#     logging.append("Fast R/W IO  = " + str(stats_n_fast_io) + "  ("+calc_percent(stats_n_fast_io,stats_n_labeled)+" out of " + str(stats_n_labeled) + ")")
#     logging.append("Slow R/W IO  = " + str(stats_n_slow_io) + "  ("+calc_percent(stats_n_slow_io,stats_n_labeled)+" out of " + str(stats_n_labeled) + ")")
    
#     stats_percent_fast_read = calc_percent(stats_n_fast_read_io,stats_n_read_io_labeled,0)
#     stats_percent_slow_read = calc_percent(stats_n_slow_read_io,stats_n_read_io_labeled,0)
#     logging.append("Fast Read-IO = " + str(stats_n_fast_read_io) + "  ("+stats_percent_fast_read+" out of " + str(stats_n_read_io_labeled) + ")")
#     logging.append("Slow Read-IO = " + str(stats_n_slow_read_io) + "  ("+stats_percent_slow_read+" out of " + str(stats_n_read_io_labeled) + ")")
#     logging.append("CDF gain     = " + str(stats_cdf_gain))
    
#     outfile_path = os.path.join(output_dir, "profile_v1.stats")
#     write_stats(outfile_path, "\n".join(logging))

# 11. Draw best-case CDF 
    # figure_path = os.path.join(output_dir, "profile_v1.lat_cdf.png")
    # extra_info = "[Outlier = " + stats_percent_outlier + "; " + stats_percent_fast_read + " fast and " + stats_percent_slow_read +  " slow; CDF gain = " + stats_cdf_gain + "]"
    # plot_raw_vs_best(figure_path, y_raw, y_best, extra_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", help="Directory path to find the trace profiles",type=str)
    parser.add_argument("-pattern", help="Pattern to match to the profile name",type=str)
    parser.add_argument("-file", help="File path of the trace profiles",type=str)
    parser.add_argument("-files", help="Arr of file path of the trace profiles", nargs='+',type=str)
    parser.add_argument("-output", help="output file name",type=str)
    parser.add_argument("-device", help="filter out the device we want",type=str)
    args = parser.parse_args()
    if (not args.file and not args.files and not (args.dir and args.pattern)):
        print("    ERROR: You must provide these arguments: -file <the input trace> ")
        exit(-1)

    trace_profiles = []
    device = args.device
    output = args.output
    if args.files:
        trace_profiles += args.files
    elif args.file:
        trace_profiles.append(args.file)
    print("trace_profiles = " + str(trace_profiles))
    
    for profile_path in trace_profiles:
        print("\nProcessing " + str(profile_path))
        start_processing(profile_path,output,device)
# How to run:
# ./tail_v1.py -file ../../data/trace_profile/nvme0n1/alibaba.cut.per_50k.most_thpt_size_iops_rand.141.trace