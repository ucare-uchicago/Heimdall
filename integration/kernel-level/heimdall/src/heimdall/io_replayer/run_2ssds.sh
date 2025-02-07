#!/bin/bash

set -e
set -u
set -o pipefail

red="\e[31m"
reset="\e[0m"


if [ $# -ne 5 ]
  then
    echo -e "${red}✘ Error: format is sudo ./replayer <algorithm> 2ssds 2 <device0>-<device1> <trace_to_device0> <trace_to_device1> ${reset}"
fi

# $1: algorithm: e.g. heimdall
# $2: trace_to_device0
# $3: trace_to_device1
# $4: device0
# $5: device1

sudo ./replayer $1 2ssds 2 $4-$5 $2 $3
