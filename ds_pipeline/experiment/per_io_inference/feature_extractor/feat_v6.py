#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import argparse

sys.path.append('../../../script')
import pattern_checker

N_HISTORY = 3

# save to a file
def write_to_file(df, filePath, has_header=True):
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

def append_prev_feature(df, num, colname):
    for i in range(1, num + 1):
        df['prev_'+ colname + '_' + str(i)] = df[colname].shift(i).values

def append_queue_len(latency, ts_submit):
    queue_process = []
    queue_len = []
    for i in range(len(ts_submit)):
        while queue_process and queue_process[0] < ts_submit[i]:
            queue_process.pop(0)
        queue_process.append(ts_submit[i])
        queue_process.sort()
        queue_len.append(len(queue_process))
    return queue_len

def start_processing(input_file):

    #    ts_record  latency  io_type   size    offset  ts_submit  size_after_replay  reject
    df = pd.read_csv(input_file)
    df['queue_len'] = append_queue_len(df['latency'].tolist(), df['ts_submit'].tolist())

    # Drop unnecessary columns
    try:
        df = df.drop(columns=["ts_record", "offset", "ts_submit", "size_after_replay"], axis=1)
    except:
        df = df.drop(columns=["ts_record", "offset", "ts_submit"], axis=1)

    # Calculate per-IO throughput
    df['throughput'] = df['size']/df['latency']
    df['throughput'] = df['throughput'].round(0)
    
    # Append Historical data
    append_prev_feature(df, N_HISTORY, 'queue_len')
    append_prev_feature(df, N_HISTORY, 'latency')
    append_prev_feature(df, N_HISTORY, 'throughput')

    # Drop the first few IOs that don't have a complete historical data
    df.drop(df.head(N_HISTORY).index, inplace=True)
    print("Removed " + str(N_HISTORY) + " first IOs because they don't have enough historical data")

    # Calculate latency increase
    df["latency_increase"] = (df["latency"]/df["prev_latency_1"]).round(2)
    
    # Calculate throughput drop
    df["throughput_drop"] = (df["prev_throughput_1"]/(df["throughput"] + 0.1)).round(2)

    # Remove any latency-related feature, except the historical value and the "latency" column
    # The latency column is needed for drawing CDF latency, not to be used as input feature
    df = df.drop(columns=["throughput", "throughput_drop", "latency_increase"], axis=1)

    # Put non_input_feature column at the last
    non_input_feature = ['latency', 'reject']
    input_features = [col for col in df.columns if col not in non_input_feature]
    df = df[input_features + non_input_feature]

    # Label all Write IO as non rejectable
    df["reject"] = df.apply (lambda row: int(0) if row['io_type'] == 0 else int(row['reject']) , axis=1)

    profile_name = os.path.basename(input_file)
    parent_dir_name = Path(input_file).parent
    profile_name = str(Path(profile_name).with_suffix('') ) # remove .trace extension
    outfile_path = os.path.join(parent_dir_name, profile_name + ".feat_v6.dataset")
    write_to_file(df, outfile_path, True)

    # Create dataset without Write IO
    df = df[df['io_type'] == 1]
    df.drop(['io_type'], axis=1, inplace=True)
    outfile_path = os.path.join(parent_dir_name, profile_name + ".feat_v6.readonly.dataset")
    write_to_file(df, outfile_path, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="File path of the labeled trace profiles",type=str)
    parser.add_argument("-files", help="Arr of file path of the labeled trace profiles", nargs='+',type=str)
    args = parser.parse_args()
    if (not args.file and not args.files):
        print("    ERROR: You must provide these arguments: -file <the labeled trace> ")
        exit(-1)
    
    trace_profiles = []
    if args.files:
        trace_profiles += args.files
    elif args.file:
        trace_profiles.append(args.file)
    print("trace_profiles = " + str(trace_profiles))
    
    for profile_path in trace_profiles:
        print("\nProcessing " + str(profile_path))
        start_processing(profile_path)
        