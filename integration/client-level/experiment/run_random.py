#!/usr/bin/env python3

import argparse
import numpy as np
import os
from pathlib import Path
import pandas as pd 
import sys
import math
import subprocess
from typing import List
import re

import subprocess
from concurrent.futures import ThreadPoolExecutor

ALGORITHM = "random"


def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(filePath, df):
    df.to_csv(filePath, header=False, index=False, sep=' ')
    print("===== output file : " + filePath)

def get_output_dir(trace_dir, devices):
    dev_names = []
    # get device name
    for dev_path in devices:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)
    output_dir = os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)
    return output_dir

def write_stats(statistics, output_file):
    with open(output_file, "w") as text_file:
        for line in statistics:
            text_file.write(str(line) + "\n")
    print("===== output file : " + output_file)

def read_raw_file(input_file):
    df = pd.read_csv(input_file, header=None, sep=' ')
    # Make sure it has 5 columns
    assert 5 == df.shape[1]
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)
    df.columns = ["ts_record","dev_num","offset","size","io_type"]
    return df
    

def run_command(command):
    try:
        # Execute the Bash command using subprocess
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")

def get_duration_from_trace(trace_path):
    with open(trace_path) as f:
        for line in f:
            if "Duration" in line:
                value_raw = line.split("=")[2]
                if "." in value_raw:
                    duration = re.findall("-?\d+\.\d+", value_raw)[0]
                else:
                    duration = re.findall("-?\d+", value_raw)[0]
                return duration


def start_processing(trace_dir, args):
    print("Processing " + str(trace_dir))
    dev_names = []
    # get device name
    for dev_path in args.devices:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)

    output_dir = os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)
    # print(output_dir)

    # Prepare the commands to run in parallel
    commands = []

    # Prepare the devices list
    devices_list_str = ""
    for device in args.devices:
        devices_list_str = devices_list_str + device + "-"
    devices_list_str = devices_list_str[:-1] # remove the final "-"
    print("The devices_list_str is {}".format(devices_list_str))

    for idx, _ in enumerate(args.devices):
        cmd = "echo 'Starting client ' " + str(idx) + "; "
        cmd += "cd " + ALGORITHM + "/; "
        trace_name = "trace_" + str(idx+1) + ".trace"
        stats_name = "trace_" + str(idx+1) + ".stats"
        trace_path = os.path.join(trace_dir, trace_name)
        stats_path = os.path.join(trace_dir, stats_name)
        duration = get_duration_from_trace(stats_path)
        # executing the replay.sh script
        # cmd += "sudo ./replay.sh -user $USER -original_device_index " + str(idx) + " -device_list " + devices_list_str + " -trace " + trace_path + " -output_dir " + output_dir + " > /dev/null 2>&1; exit"
        cmd += "sudo ./replay.sh -user $USER -original_device_index " + str(idx) + " -device_list " + devices_list_str + " -trace " + trace_path + " -output_dir " + output_dir + " -duration " + duration + "; exit"
        # cmd += "echo 'Done running client 1'"
        # print("cmd = " + cmd)
        commands.append(cmd)
    
    # Create a thread pool to run the commands in parallel
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        # Submit each command to the thread pool
        for command in commands:
            executor.submit(run_command, command)
    print("Output dir = " + output_dir)
    subprocess.run("stty sane", shell=True, check=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-devices", help="The array of storage devices (separated by space)", nargs='+',type=str)
    parser.add_argument("-trace_dir", help="File path to the trace sections description ", type=str)
    parser.add_argument("-trace_dirs", help="Arr of file path to the trace sections description", nargs='+',type=str)
    parser.add_argument("-resume", help="Resume the processing, not replaying the replayed traces", action='store_true')

    args = parser.parse_args()
    if (not args.devices) or (not (args.trace_dir or args.trace_dirs)):
        print("    ERROR: You must provide these arguments: -devices <the array of storage devices> -trace_dir <the trace dir> or -trace_dirs <the array of trace dirs")
        exit(-1)

    trace_dirs = []
    if args.trace_dirs:
        trace_dirs += args.trace_dirs
    elif args.trace_dir:
        trace_dirs.append(args.trace_dir)

    print("trace_paths = " + str(trace_dirs))
    print("algo = " + ALGORITHM)
    print("devices = " + str(args.devices))
    print("Found " + str(len(trace_dirs)) + " trace dirs")

    
    for idx, trace_dir in enumerate(trace_dirs):
        # Print a message after the terminal reset
        print("\nProcessing trace dir " + str(idx+1) + " out of " + str(len(trace_dirs)))
        if args.resume: 
            # check if the trace is already replayed
            output_dir = get_output_dir(trace_dir, args.devices)
            output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
            # print("output_stat_path = " + output_stat_path)
            if os.path.isfile(output_stat_path):
                print("     The trace is already replayed, skipping")
                continue
        subprocess.run("stty sane", shell=True, check=True)
        start_processing(trace_dir, args)