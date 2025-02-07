#!/usr/bin/env python3

import argparse
import subprocess
from subprocess import call
from pathlib import Path

def start_processing(input_file):
    print("Process")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="Path to the dataset", type=str)
    parser.add_argument("-datasets", help="Path of the datasets", nargs='+', type=str)
    parser.add_argument("-model", help="The filename of the model (.py)",type=str)
    parser.add_argument("-train_eval_split", help="Ratio to split the dataset for training and evaluation",type=str)
    args = parser.parse_args()
    if (not args.dataset and not args.datasets and not args.model and not args.train_eval_split):
        print("    ERROR: You must provide these arguments: -dataset <the labeled trace> -model <the model name> -train_eval_split <the split ratio> ")
        exit(-1)

    arr_dataset = []
    if args.datasets:
        arr_dataset += args.datasets
    elif args.dataset:
        arr_dataset.append(args.dataset)
    print("trace_profiles = " + str(arr_dataset))
    
    for dataset_path in arr_dataset:
        print("\nTraining on " + str(dataset_path))
        command = "./" + args.model + ".py -dataset " + dataset_path + " -train_eval_split " + args.train_eval_split
        subprocess.call(command, shell=True)
