#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from os import listdir

# pattern may include '*' or '.' or '_' and valid alphabet/number
# * should not be placed in the back of the pattern
def check_pattern(pattern, filename):
    # check if the filename compy with the pattern
    pattern = pattern.replace('.','*.*') # Will treat '.' as '*'
    patterns = pattern.split("*")
    subnames = filename.split('.')
    # print(subnames, patterns)
    # check the extension
    last_pattern = patterns.pop()
    if last_pattern == "": 
        print("ERROR: Pattern ( " + pattern + " ) should NOT be ended with an \"*\" because we need the extension of the file!")
        return None
    if last_pattern in subnames.pop(): # Checking the extension
        # do further checking
        is_matched = True
        # print("prior while",subnames, patterns)
        dot_detected = False
        while patterns != [] and is_matched and subnames != []:
            cur_pattern = patterns.pop()
            # print("--- ",subnames, patterns)
            if cur_pattern == ".": 
                dot_detected = True
            else:
                if dot_detected: 
                    dot_detected = False
                    # The next pattern must be found in the next idx
                    is_matched = True if cur_pattern in subnames.pop() else False
                else:
                    while subnames != []:
                        word = subnames.pop()
                        if cur_pattern in word:
                            # print(word, sub_pattern)
                            subnames.append(word) # the next sub_pattern might match this
                            cur_pattern = None
                            break 
                    if cur_pattern != None: is_matched = False
        return True if is_matched else False 
    return False

def get_files(dir, pattern):
    # print("DIR ", dir)
    tracefiles = []
    for data_path in listdir(dir):
        if os.path.isdir(os.path.join(dir, data_path)):
            tracefiles += get_files(os.path.join(dir, data_path), pattern)
        if not data_path.startswith(".") and os.path.isfile(os.path.join(dir, data_path)):
            is_matched = check_pattern(pattern, data_path)
            if is_matched == None: return [] # Pattern is invalid
            if is_matched: tracefiles.append(os.path.join(dir, data_path))
    # print(tracefiles)
    return tracefiles

# this will allow * to be placed in the front/back of the pattern
def check_dir_pattern(str_pattern, filename):
    # check if the filename compy with the pattern
    str_pattern = str_pattern.replace('.','*') # Will treat '.' as '*'
    patterns = str_pattern.split("*")

    subnames = filename.split('.')
    sub_pattern = None
    while patterns != [] and subnames != []:
        # print(subnames, patterns)
        sub_pattern = patterns.pop()
        if len(sub_pattern) == 0:
            # this is a *
            sub_pattern = None
        else: # not a *, must be compared
            while subnames != []:
                word = subnames.pop()
                if sub_pattern in word:
                    # print(word, sub_pattern)
                    subnames.append(word) # the next sub_pattern might match this
                    sub_pattern = None
                    break 
    return True if sub_pattern == None else False

def get_dirs(dir, dir_pattern):
    # print("DIR ", dir)
    matched_dirs = []
    for data_path in listdir(dir):
        if os.path.isdir(os.path.join(dir, data_path)):
            # print(data_path)
            if check_dir_pattern(dir_pattern, data_path): 
                matched_dirs.append(os.path.join(dir, data_path))
    return matched_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pattern", help="Pattern of the filename",type=str)
    parser.add_argument("-file", help="File path of the trace",type=str)
    args = parser.parse_args()
    if (not args.file):
        print("    ERROR: You must provide these arguments: -file <the path of the file> -pattern <the targetted pattern>")
        exit(-1)
    input_path = args.file
    parent_dir = str(Path(input_path).parent)
    filename = os.path.basename(args.file)
    
    res = "True" if check_pattern(args.pattern, filename) else "False"
    print(res)