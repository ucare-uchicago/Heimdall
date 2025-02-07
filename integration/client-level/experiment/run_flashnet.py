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
import time
import re

import subprocess
from concurrent.futures import ThreadPoolExecutor

ALGORITHM = "flashnet"

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(filePath, df):
    df.to_csv(filePath, header=False, index=False, sep=' ')
    print("===== output file : " + filePath)

def write_stats(statistics, output_file):
    with open(output_file, "w") as text_file:
        for line in statistics:
            text_file.write(str(line) + "\n")
    print("===== output file : " + output_file)

def get_output_dir(trace_dir, devices):
    dev_names = []
    # get device name
    for dev_path in devices:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)
    output_dir = os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)
    return output_dir

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

def start_processing(trace_dir, args, specific_workplace):
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
        cmd += "cd " + specific_workplace + "/; "   # go the specific workplace
        trace_name = "trace_" + str(idx+1) + ".trace"
        stats_name = "trace_" + str(idx+1) + ".stats"
        trace_path = os.path.join(trace_dir, trace_name)
        stats_path = os.path.join(trace_dir, stats_name)
        duration = get_duration_from_trace(stats_path)
        
        # executing the replay.sh script
        # cmd += "sudo ./replay.sh -user $USER -original_device_index " + str(idx) + " -devices_list " + devices_list_str + " -trace " + trace_path + " -output_dir " + output_dir + " > /dev/null 2>&1; exit"
        cmd += "sudo ./replay.sh -user $USER -original_device_index " + str(idx) + " -devices_list " + devices_list_str + " -trace " + trace_path + " -output_dir " + output_dir + " -duration " + duration + "; exit"
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


def make_flashnet(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    '''
        Conduct:
            1. Copy the training weights (in header files) of flashnet to a tmp directory
            2. make flashnet
        Return:
            Whether `make` succeed or not.
    '''
    # 1.1. copy the weights model to a tmp folder
    tmp_weights_header_path = ""
    dev_names = []
    for dev_path in devices_list:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)
    path_to_weights = os.path.join(str(trace_dir), "...".join(dev_names), "{}/training_results/weights_header_2ssds".format(ALGORITHM))
    ## check whether weights header files exist or not
    for dev_id, dev_name in enumerate(dev_names):
        header_file_path = os.path.join(str(path_to_weights), "w_Trace_dev_{}.h".format(dev_id))
        print("header file path = {}".format(header_file_path))
        if not os.path.exists(header_file_path):
            print("header file: {} not exist.".format(header_file_path))
            return False
    try:
        tmp_weights_header_path = "{}/2ssds_weights_header".format(specific_workplace)
        os.makedirs(tmp_weights_header_path)
        for dev_id, dev_name in enumerate(dev_names):
            header_file_path = os.path.join(str(path_to_weights), "w_Trace_dev_{}.h".format(dev_id))
            subprocess.run(["cp", header_file_path, tmp_weights_header_path], check=True)
            print(f"File {header_file_path} copied to {tmp_weights_header_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

    # 1.2. make flashnet
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


def train_flashnet(trace_dir: str, devices: List[str]) -> bool:
    '''
        Input:
            `trace_dir`: path of a directory, which stores the result of baseline.
            `devices`: e.g. ['/dev/nvme0n1', '/dev/nvme2n1']
        Perform:
            Train flashnet model based on baseline's result.
            The training results(e.g. header file of weights) will be stored under the given `trace_dir`
        Output:
            Flag about whether the training is successful or not.
            True: success. Flase: fail.

        ./train_flashnet.sh nvme0n1 nvme2n1 $FLASHNET_INTEGRATION/data/grouping_2_traces_v1.duration_2.5_mins/msr.cut.per_5mins.rw_51_49.774/msr.cut.per_5mins.rw_71_29.565/modified.rerate_2.00...modified.rerate_2.00
    '''
    dev_names = []
    for dev_path in devices:
        dev_name = os.path.basename(dev_path)
        dev_names.append(dev_name)
    # 1. format training script
    train_command = ["./train_flashnet.sh"]
    # 2. format devices
    for dev in dev_names:
        train_command.append(dev)
    # 3. format input dir
    if os.path.exists(trace_dir):
        train_command.append(trace_dir)
    else:
        print("[Error!!] trace_dir not exist: {}".format(trace_dir))
        return False

    print("training command: {}".format(train_command))
    # Run the train command
    original_directory = os.getcwd()
    try:
        os.chdir("./{}".format(ALGORITHM))
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as train_e:
        print("Error running 'training' {}".format(train_e))
        os.chdir(original_directory)
        return False

    os.chdir(original_directory)
    return True
    
def do_sleep(timer_mins, time_start):
    time_end = pd.Timestamp.now()
    time_elapsed = (time_end - time_start).seconds
    print("time_elapsed = " + str(time_elapsed))
    if time_elapsed < args.timer_mins * 60:
        print("Sleeping for " + str(args.timer_mins * 60 - time_elapsed) + " seconds")
        time.sleep(args.timer_mins * 60 - time_elapsed)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-devices", help="The array of storage devices (separated by space)", nargs='+',type=str)
    parser.add_argument("-trace_dir", help="File path to the trace sections description ", type=str)
    parser.add_argument("-trace_dirs", help="Arr of file path to the trace sections description", nargs='+',type=str)
    parser.add_argument("-resume", help="Resume the processing, not replaying the replayed traces", action='store_true')
    parser.add_argument("-timer_mins", help="Will not replay the traces until it reaches this timer target", type=int, default=0)
    parser.add_argument("-only_training", help="Only do training", action='store_true', default=False)
    parser.add_argument("-only_replaying", help="Only do replaying, assuming that it's already replayed", action='store_true', default=False)
    parser.add_argument("-if_model_updated", help="Will check if the model weights is newer than the replayed traces", action='store_true', default=False)
    parser.add_argument("-reverse", help="Will start from the last combination", action='store_true')
    
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
    print("devices = " + str(args.devices))
    print("algo = " + ALGORITHM)
    print("Found " + str(len(trace_dirs)) + " trace dirs")

    if args.reverse:
        trace_dirs = trace_dirs[::-1]
        print("Reversed the order of the trace dirs")

    for idx, trace_dir in enumerate(trace_dirs):

        print("\nProcessing trace dir " + str(idx+1) + " out of " + str(len(trace_dirs)))
        output_dir = get_output_dir(trace_dir, args.devices)
        output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
        output_weights_path_0 = os.path.join(output_dir, "training_results", "weights_header_2ssds", "w_Trace_dev_0.h")
        output_weights_path_1 = os.path.join(output_dir, "training_results", "weights_header_2ssds", "w_Trace_dev_1.h")

        if args.resume: 
            # check if the trace is already replayed
            if os.path.isfile(output_stat_path) and os.path.isfile(output_weights_path_0) and os.path.isfile(output_weights_path_1):
                print("     The trace is already trained and replayed, skipping")
                continue

        time_start = pd.Timestamp.now()

        if not args.only_replaying:
            # will skip if the training is already done
            if args.resume: 
                # check if it can find the header files
                if os.path.isfile(output_weights_path_0) and os.path.isfile(output_weights_path_1):
                    print("     The model is trained, skipping\n\n")
                    continue

            # 1. train flashnet
            train_result = train_flashnet(trace_dir, args.devices)
            if train_result == False:   # Fail in training
                print("\n[Train Flashnet Error!!!!], dir: {}".format(trace_dir))
                exit(-1)

        if args.timer_mins > 0:
            # will start the replaying after the timer target
            # this is waiting for another replayer to finish
            do_sleep(args.timer_mins, time_start)

        if args.only_training:
            # avoid replaying
            continue

        # check if the model is ready
        if not os.path.isfile(output_weights_path_0) or not os.path.isfile(output_weights_path_1):
            print("     WARNING: The model weights are not ready (not-done training), skipping\n\n")
            continue

        # check if it has the updated model weights
        if os.path.isfile(output_stat_path) and args.if_model_updated:
            # check the last modified time of the model weights vs the replayed traces (if any)
            weight_modified_time = os.path.getmtime(output_weights_path_0)
            stat_modified_time = os.path.getmtime(output_stat_path)
            if weight_modified_time < stat_modified_time:
                print("     The model is OLDER than the replayed traces, skipping\n\n")
                continue

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

        # 2. make flashnet
        make_result = make_flashnet(trace_dir, args.devices, specific_workplace)
        if make_result == False:   # fail to make
            print("\n[Make ERROR: Weights is not complete. Please (re)train the model on this trace], dir: {}".format(trace_dir))
            exit(-1)

        # 3. replaying traces with help of flashnet
        subprocess.run("stty sane", shell=True, check=True)
        start_processing(trace_dir, args, specific_workplace)

        # 4. Delete the tmp workplace after replaying
        if delete_dir(specific_workplace) == False:
            print("Workplace delete error: {}".format(specific_workplace))
            exit(-1)