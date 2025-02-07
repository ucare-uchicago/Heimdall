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
import math

sys.path.append('../../../script')
import default_ip_finder

# IF more than 20% of individual IO is being rejected, the per_batch label will be reject!
MIN_REJECTION_COUNT = 0.50 # 0.2 == 20%

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
    df = pd.read_csv(input_file, sep=',')

    # Make sure that it has "reject" label
    assert "reject" in list(df.columns)
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)

    # [NO NEED] filter: remove io that doesn't executed properly (can't read/write all bytes)
    # df = df[df['size'] == df['size_after_replay']]
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

def append_prev_feature(df, num, col_target):
    colnames = []
    for i in range(0, num):
        colname = col_target + '_' + str(i)
        df[colname] = df[col_target].shift(i).values
        colnames.append(colname)
    # print(df.head(20))
    return colnames

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
    plt.title('CDF of Read-only IOs with MIN_REJECTION_COUNT = ' + calc_percent(MIN_REJECTION_COUNT,1) + ' \n' + extra_info)
    p70_lat = np.percentile(y_raw, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw latency",  color="red")
    plt.plot(x_1, y_1, label = "FlashNet-best-case (batched)", linestyle='dashdot', color="green")
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
    
# 3. Calculate the AUC
    cdf_raw = auc(x_2, y_2)
    cdf_best = auc(x_1, y_1)
    # print(len(y_raw), len(x_2))
    # print(len(y_best), len(x_1))

    # print('Raw CDF area  : {}'.format(cdf_raw))
    # print('Best CDF area : {}'.format(cdf_best))
    percent_gain = calc_percent(cdf_best - cdf_raw, cdf_raw)
    # print(percent_gain)

    return percent_gain
    # Note: Do not use np.trapz(xx,yy), it doesn't calculate valid area

def get_write_ios(batched_io_type, batched_latency):
    res = []
    for idx, io_type in enumerate(batched_io_type):
        if io_type == 0: res.append(batched_latency[idx])
    return res

def start_processing(input_path, batch_size): 
# 0. Read labeled data from per_io dataset
    df = read_file(input_path)
    stats_total_io = len(df)
    if 'readonly' not in input_path:
        stats_n_read = len(df[df["io_type"] == 1])
    else:
        stats_n_read = len(df)
  
    # Rename "reject" col to "per_io_reject"
    cols = list(df.columns)
    df["per_io_reject"] = df["reject"]
    # drop the original reject column
    df = df.drop('reject', axis=1)

# 1. Convert all write IO to non reject [We can't reject the write IO]
    if 'readonly' not in input_path:
        df["per_io_reject"] = df.apply(lambda row: int(0) if row['io_type'] == 0 else int(row['per_io_reject']), axis=1)

# 2. Form the perbatch data by horizontally aggregating historical data into single batch row
    cols = list(df.columns)
    col_names_batched_data = []

    # Column from feat_v6
        # Index(['io_type', 'size', 'queue_len', 'prev_queue_len_1', 'prev_queue_len_2',
        #    'prev_queue_len_3', 'prev_latency_1', 'prev_latency_2',
        #    'prev_latency_3', 'prev_throughput_1', 'prev_throughput_2',
        #    'prev_throughput_3', 'latency', 'per_io_reject'],
        #   dtype='object')


    # column that will be aggregated horizontally
    cols_readonly = ['size','latency', 'per_io_reject']
    if 'readonly' not in input_path:
        cols_readwrite = ['io_type', 'size','latency', 'per_io_reject']
    else:
        cols_readwrite = ['size','latency', 'per_io_reject']
    cols_selected = cols_readonly if 'readonly' in input_path else cols_readwrite
    for col in cols_selected:
        colnames = append_prev_feature(df, batch_size, col)
        col_names_batched_data += colnames
    
    # adding column that we will use as it is
    col_names_batched_data += [ 'queue_len', 'prev_queue_len_1', 'prev_queue_len_2', 'prev_queue_len_3', 'prev_latency_1', 'prev_latency_2', 'prev_latency_3', 'prev_throughput_1', 'prev_throughput_2','prev_throughput_3']

# 3. Assign per_batch labeling
    cols_to_sum = [col for col in df.columns if col.startswith("per_io_reject_")]
    
    df['tot_rejections'] = df[cols_to_sum].sum(axis=1)

    # Based on "batched_per_io_reject"
    min_reject_count = math.ceil(MIN_REJECTION_COUNT * batch_size)
    df['per_batch_reject'] = df.apply(lambda row: 1 if row['tot_rejections'] >= min_reject_count else 0, axis=1)
    # print(df.head(10))

# 4. Generate the index that store complete per-batch data
    # the first 5 (batch_size = 5) IO is useless because it doesn't have enough historical value
    # df_per_batch is the subset of df
    batch_idxs = range(0 + batch_size,len(df),batch_size)
    df_per_batch = df.iloc[batch_idxs]
    col_names_batched_data.append("per_batch_reject") # we also need this column for our per batch dataset

    # drop irrelevant columns
    df_per_batch = df_per_batch.drop(columns=[col for col in df_per_batch.columns if col not in col_names_batched_data], axis=1)
    df_per_batch.reset_index(drop=True, inplace=True)

# 4. Write the labeled data
    batch_analyzer_name = str(Path(os.path.basename(__file__)).with_suffix(''))
    stats_n_labeled = len(df_per_batch)
    print("#batch IO labeled = " + str(stats_n_labeled))

    profile_name = os.path.basename(input_path)
    parent_dir_name = os.path.basename(Path(input_path).parent)
    storage_device_name = os.path.basename(Path(input_path).parent.parent)

    profile_name = str(Path(profile_name).with_suffix('') ) # remove .labeled extension
    output_dir = "../../../data/dataset/" + storage_device_name + "/" + parent_dir_name + "/"
    create_output_dir(output_dir)
    batch_labeled_filename = profile_name + "."  + "batch_" + str(batch_size)
    outfile_path = os.path.join(output_dir, batch_labeled_filename + ".dataset")
    
    write_to_file(df_per_batch, outfile_path, True)

    stats_n_fast_batch = len(df_per_batch[df_per_batch['per_batch_reject'] == 0])
    stats_n_slow_batch = len(df_per_batch[df_per_batch['per_batch_reject'] == 1])
    print("Fast batch = " + str(stats_n_fast_batch))
    print("Slow batch = " + str(stats_n_slow_batch))

# 5. Calculate the CDF gain on Read-only IO
    # Remove the write IO
    if 'readonly' not in input_path:
        df = df[df['io_type'] == 1]

    # get all latency from the accepted batch
    y_best = list(df.loc[df['per_batch_reject'] == 0, 'latency'].values)
    y_raw = list(df['latency'].values)

    stats_cdf_gain = calc_cdf_gain( y_raw, y_best)
    # print(stats_cdf_gain)

# 5. Write the stats
    logging = []
    logging.append("============================================")
    logging.append("                BASIC INFO ")
    logging.append("============================================")
    logging.append("Labeled Profile name = " + profile_name)
    logging.append("Full path    = " + input_path)
    stats_read_ratio = int((stats_n_read/stats_total_io)*100)
    logging.append("R:W ratio    = " + str(stats_read_ratio) + ":" + str(100-stats_read_ratio))
    logging.append("#IO          = " + str(stats_total_io))
    logging.append("#writes      = " + str(stats_total_io - stats_n_read))
    logging.append("#reads       = " + str(stats_n_read))
    logging.append("============================================")
    logging.append("                STATISTICS")
    logging.append("============================================")
    logging.append("Min #IO_reject to reject the batch = " + calc_percent(MIN_REJECTION_COUNT, 1))
    logging.append("Batch size         = " + str(batch_size))
    logging.append("#Batch IO labeled  = " + str(stats_n_labeled))
    stats_percent_fast_batch = calc_percent(stats_n_fast_batch,stats_n_labeled)
    stats_percent_slow_batch = calc_percent(stats_n_slow_batch,stats_n_labeled)
    logging.append("Fast batch  = " + str(stats_n_fast_batch) + "  ("+ stats_percent_fast_batch +" out of " + str(stats_n_labeled) + ")")
    logging.append("Slow batch  = " + str(stats_n_slow_batch) + "  ("+ stats_percent_slow_batch +" out of " + str(stats_n_labeled) + ")")
    
    logging.append("CDF gain     = " + str(stats_cdf_gain))
    
    outfile_path = os.path.join(output_dir, batch_labeled_filename + ".stats")
    write_stats(outfile_path, "\n".join(logging))

# 11. Draw best-case CDF 
    figure_path = os.path.join(output_dir, batch_labeled_filename + "_" + str(int(MIN_REJECTION_COUNT*100)) + ".lat_cdf.png")
    extra_info = "[Batch size = " + str(batch_size) + "; fast = " + stats_percent_fast_batch + "; slow = " + stats_percent_slow_batch +  "; CDF gain = " + stats_cdf_gain + "]"
    plot_raw_vs_best(figure_path, y_raw, y_best, extra_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", help="Number of IO per batch",type=int)
    parser.add_argument("-file", help="File path of the trace profiles",type=str)
    parser.add_argument("-files", help="Arr of file path of the trace profiles", nargs='+',type=str)
    args = parser.parse_args()
    if ( (not args.file or not args.files) and not args.batch_size):
        print("    ERROR: You must provide these arguments: -batch_size <#IO per batch> -file <the input trace> ")
        exit(-1)

    trace_profiles = []
    if args.files:
        trace_profiles += args.files
    elif args.file:
        trace_profiles.append(args.file)
    print("trace_profiles = " + str(trace_profiles))
    
    for profile_path in trace_profiles:
        print("\nProcessing " + str(profile_path))
        start_processing(profile_path, args.batch_size)
# How to run:
# ./batch_for_feat_v6.py -file ../../data/trace_profile/nvme0n1/alibaba.cut.per_50k.most_thpt_size_iops_rand.141.trace