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
import shutil
import re

import subprocess
from concurrent.futures import ThreadPoolExecutor


ALGORITHM = "linnos_hedging"
Inflection_percentile = 90


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

def read_file(input_file):
    df = pd.read_csv(input_file, header=None, sep=' ')
    # Make sure it has 5 columns
    assert 5 == df.shape[1]
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)
    df.columns = ["ts_record","dev_num","offset","size","io_type"]
    return df


def parse_hedging_latency(percentile: float, trace_dir: str, devices: List[str]) -> List[float]:
    '''
        Given a trace_dir, devices, and percentile being used to replay baseline traces.
        1. Retrieve the result of baseline
        2. Get the value at specifiy percentile (e.g. the read latency at p95)
    '''
    # 1. Check whether the percentile is valid. (e.g. We dont have value at 101%)
    if percentile <= 0 or percentile >= 100:
        print("\n[Error] percentile is {}, please specify a percentile that is between 0 and 100".format(percentile))
        exit(-1)

    dev_names = []
    # 2. Retrieve the results of baseline
    for dev_path in devices:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)
    baseline_result_dir = os.path.join(str(trace_dir), "...".join(dev_names), "baseline")

    # 3. For each device, retrieve the value at the specified percentile
    percentile_value_list = []
    for i in range(1, len(devices) + 1):
        result_path = os.path.join(str(baseline_result_dir), "trace_{}.trace".format(i))
        read_io_lat_list = []
        # check whether the result of baseline exists
        if not os.path.exists(result_path):
            print("\n[Error!] Result of Baseline Not Exist")
            exit(-1)
        with open(result_path, 'r') as f:
            for line in f:
                tok = list(map(str.strip, line.split(",")))
                if int(tok[2]) == 1:   # read
                    read_io_lat_list.append(int(tok[1]))   # [latency, IO_type]
        # Calculate the specified percentile
        percentile_value = np.percentile(read_io_lat_list, percentile)
        percentile_value_list.append(percentile_value)
    print("percentile_value_list: {}".format(percentile_value_list))
    
    return percentile_value_list[:]


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

def start_processing(trace_dir, args, hedging_latency_list, specific_workplace):
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

    # client 0 will write to device 0
    # client 1 will write to device 1
    for idx, device in enumerate(args.devices):
        cmd = "echo 'Starting client ' " + str(idx) + "; "
        cmd += "cd " + specific_workplace + "/; "
        trace_name = "trace_" + str(idx+1) + ".trace"
        stats_name = "trace_" + str(idx+1) + ".stats"
        trace_path = os.path.join(trace_dir, trace_name)
        stats_path = os.path.join(trace_dir, stats_name)
        duration = get_duration_from_trace(stats_path)
        print("hedging after: {} us".format(hedging_latency_list[idx]))
        # executing the replay.sh script
        # cmd += "sudo ./replay.sh -user $USER -original_device_index " + str(idx) + " -devices_list " + devices_list_str + " -trace " + trace_path + " -output_dir " + output_dir + " > /dev/null 2>&1; exit"
        cmd += "sudo ./replay.sh -user $USER -original_device_index " + str(idx) + " -devices_list " + devices_list_str + " -hedging_latency " + str(hedging_latency_list[idx]) + " -trace " + trace_path + " -output_dir " + output_dir + " -duration " + duration + "; exit"
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


def delete_dir(path: str) -> bool:
    '''
        Given a directory path, try to delete it.
        If Succeed in deleting it, return True. Otherwise, return False.
    '''
    try:
        shutil.rmtree(path)
        print(f"Directory '{path}' has been deleted.")
    except OSError as e:
        print(f"Error: {e}")
        return False
    return True


def make_linnos(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    '''
        Conduct:
            1. Copy the training weights (in header files) of linnos to a tmp directory
            2. make linnos
        Return:
            Whether `make` succeed or not.
    '''
    # 1.1. copy the weights model to a tmp folder
    tmp_weights_header_path = ""
    dev_names = []
    for dev_path in devices_list:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)
    # directly use the training result of linnos (Without hedging)
    path_to_weights = os.path.join(str(trace_dir), "...".join(dev_names), "linnos/training_results/weights_header_2ssds")
    ## check whether weights header files exist or not
    for dev_id, dev in enumerate(dev_names):
        header_file_path = os.path.join(str(path_to_weights), "w_Trace_dev_{}.h".format(dev_id))
        print("header file path = {}".format(header_file_path))
        if not os.path.exists(header_file_path):
            print("header file: {} not exist. Please Run Experiment of Linnos(with out hedging) to get the training result.".format(header_file_path))
            return False
    try:
        tmp_weights_header_path = "{}/2ssds_weights_header".format(specific_workplace)
        os.makedirs(tmp_weights_header_path)
        for dev_id, dev in enumerate(dev_names):
            header_file_path = os.path.join(str(path_to_weights), "w_Trace_dev_{}.h".format(dev_id))
            subprocess.run(["cp", header_file_path, tmp_weights_header_path], check=True)
            print(f"File {header_file_path} copied to {tmp_weights_header_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

    # 1.2. make linnos
    original_directory = os.getcwd()
    try:
        os.chdir("{}".format(specific_workplace))
        # Run the "make" command in the new working directory
        try:
            subprocess.run(["make"], check=True)
        except subprocess.CalledProcessError as make_error:
            print(f"Error running 'make': {make_error}")

            print("Removing any .o files in the /tmp folder")
            # Remove any .o files in the /tmp folder
            for file in Path("/tmp").glob("*.o"):
                file.unlink()
            
            print("Retrying to run 'make'")
            subprocess.run(["make"], check=True)

            # os.chdir(original_directory)
            # return False
        os.chdir(original_directory)
    except FileNotFoundError:
        print(f"Directory '{specific_workplace}' not found.")
        os.chdir(original_directory)
        return False

    return True


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-devices", help="The array of storage devices (separated by space)", nargs='+',type=str)
    parser.add_argument("-trace_dir", help="File path to the trace sections description ", type=str)
    parser.add_argument("-resume", help="Resume the processing, not replaying the replayed traces", action='store_true')
    parser.add_argument("-trace_dirs", help="Arr of file path to the trace sections description", nargs='+',type=str)
    parser.add_argument("-hedging_percentile", help="Hedging the IO after p-th", type=str)

    args = parser.parse_args()
    if (not args.devices) or (not (args.trace_dir or args.trace_dirs))or (not args.hedging_percentile):
        print("    ERROR: You must provide these arguments: -devices <the array of storage devices> -trace_dir <the trace dir> or -trace_dirs <the array of trace dirs> -hedging_percentile <percentile, default 98>")
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
    print("Hedging the IO after p" + str(args.hedging_percentile))

    for idx, trace_dir in enumerate(trace_dirs):

        if args.resume: 
            # check if the trace is already replayed
            output_dir = get_output_dir(trace_dir, args.devices)
            output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
            # print("output_stat_path = " + output_stat_path)
            if os.path.isfile(output_stat_path):
                print("     The trace is already replayed, skipping")
                continue

        # 1.[Training] We don't train the Linnos module again for Linnos+Hedging, but directly use the training result of Linnos(without hedging)

        # [NEW WORKPLACE] To avoid conflict make, we copy the source code to a tmp file, specified with current devices name
        specific_workplace = "./tmp_running/{}_{}...{}".format(ALGORITHM, args.devices[0].split("/")[2], args.devices[1].split("/")[2])
        if os.path.exists(specific_workplace):
            if delete_dir(specific_workplace) == False:
                print("Workplace delete error: {}".format(specific_workplace))
                exit(-1)
        print("The specific workplace is {}".format(specific_workplace))
        try:
            subprocess.run(["cp", "-r", "./{}".format(ALGORITHM), specific_workplace], check=True)
        except:
            print("cp workplace wrong!")
            exit(-1)

        # 2. make linnos_hedging
        make_result = make_linnos(trace_dir, args.devices, specific_workplace)
        if make_result == False:   # fail to make
            print("\n[Make Error!!!!], dir: {}".format(trace_dir))
            exit(-1)
    
        # 3. replaying traces with help of linnos
        print("\nProcessing trace dir " + str(idx+1) + " out of " + str(len(trace_dirs)))
        subprocess.run("stty sane", shell=True, check=True)

        # First step for Hedging: get the estimated ratio of redirecting IO to each device
        hedging_latency = parse_hedging_latency(float(args.hedging_percentile), trace_dir, args.devices)
        start_processing(trace_dir, args, hedging_latency, specific_workplace)

        # 4. Delete the tmp workplace after replaying
        if delete_dir(specific_workplace) == False:
            print("Workplace delete error: {}".format(specific_workplace))
            exit(-1)
