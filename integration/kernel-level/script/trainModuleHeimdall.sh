#!/bin/bash

set -e
set -u
set -o pipefail

green="\e[32m"
red="\e[31m"
reset="\e[0m"

SCRIPT_DIR=$(pwd)

# ================================== 1. Read the Config and Validate =========================================
# Source the configuration file
source $HEIMDALL_KERNEL/config/config.conf

# Validate traces
if [ ! -f "$TRACE_TO_SSD_DEVICE0" ]; then
    echo -e "${red}✘ Error: Trace not exist, please specify it in $HEIMDALL_KERNEL/config/config.conf: $TRACE_TO_SSD_DEVICE0 ${reset}"
    exit 1
fi

if [ ! -f "$TRACE_TO_SSD_DEVICE1" ]; then
    echo -e "${red}✘ Error: Trace not exist, please specify it in $HEIMDALL_KERNEL/config/config.conf: $TRACE_TO_SSD_DEVICE1 ${reset}"
    exit 1
fi
echo -e "${green}✔ Input traces valid. ${reset}"

# Validate SSD devices

ABBREVIATE_SSD_DEVICE0="${SSD_DEVICE0#/dev/}"
ABBREVIATE_SSD_DEVICE1="${SSD_DEVICE1#/dev/}"

is_ssd() {
  local device="$1"
  local rotational=$(lsblk -no rota "/dev/$device" 2>/dev/null)
  if [[ "$rotational" =~ "0" ]]; then
    return 0  # It's an SSD
  else
    return 1  # It's not an SSD
  fi
}

# Get a list of all unmounted block devices with no mounted partitions
unmounted_devices=($(lsblk -o name,mountpoint -n | awk '$2 == "" {print $1}' | grep -E '^[a-zA-Z0-9]+$'))
unmounted_ssd_names=""
for device in "${unmounted_devices[@]}"; do
  if is_ssd "$device"; then
    # Check if there are no mounted partitions for this device
    if ! mount | grep -q "/dev/$device"; then
      unmounted_ssd_names="$unmounted_ssd_names $device"
    fi
  fi
done

if [[ ! $unmounted_ssd_names == *$ABBREVIATE_SSD_DEVICE0* ]]; then
    echo -e "${red}✘ Error: Device $SSD_DEVICE0 not valid. Possible reasons: 1. Not Exist. 2. Not a SSD. 3. Not unmounted. ${reset}"
    exit 1
fi

if [[ ! $unmounted_ssd_names == *$ABBREVIATE_SSD_DEVICE1* ]]; then
    echo -e "${red}✘ Error: Device $SSD_DEVICE1 not valid. Possible reasons: 1. Not Exist. 2. Not a SSD. 3. Not unmounted. ${reset}"
    exit 1
fi
echo -e "${green}✔ SSDs valid. ${reset}"

echo "SSD_DEVICE0: $SSD_DEVICE0"
echo "SSD_DEVICE1: $SSD_DEVICE1"
echo "TRACE_TO_SSD_DEVICE0: $TRACE_TO_SSD_DEVICE0"
echo "TRACE_TO_SSD_DEVICE1: $TRACE_TO_SSD_DEVICE1"

# format a output directory
ABBREVIATE_TRACE_TO_DDS_DEVICE0=$(echo "$TRACE_TO_SSD_DEVICE0" | rev | cut -d'/' -f1 | cut -d'.' -f2- | rev)
ABBREVIATE_TRACE_TO_DDS_DEVICE1=$(echo "$TRACE_TO_SSD_DEVICE1" | rev | cut -d'/' -f1 | cut -d'.' -f2- | rev)
OUTPUT_DIR="$HEIMDALL_KERNEL/benchmark/results/${ABBREVIATE_TRACE_TO_DDS_DEVICE0}_${ABBREVIATE_TRACE_TO_DDS_DEVICE1}"
mkdir -p $OUTPUT_DIR
cp $HEIMDALL_KERNEL/config/config.conf $OUTPUT_DIR/config_copy.conf   # record the config 

# ================================== 2. Train module =========================================
echo "Module Training..."
cd $HEIMDALL_KERNEL/heimdall/src/heimdall/training
touch training_results.txt
./train.sh $TRACE_TO_SSD_DEVICE0 $TRACE_TO_SSD_DEVICE1 $SSD_DEVICE0 $SSD_DEVICE1 > training_results.txt 2>&1
cd $SCRIPT_DIR

# store the trained weights
rm -rf $OUTPUT_DIR/heimdall
mkdir -p $OUTPUT_DIR/heimdall
mv $HEIMDALL_KERNEL/heimdall/src/heimdall/training/weights_header_2ssds/* $OUTPUT_DIR/heimdall/

echo -e "${green}✔ Finish Module Training. ${reset}"
echo "Trained weights stored in: $OUTPUT_DIR/heimdall/"
echo "Log of training process stored in: $HEIMDALL_KERNEL/heimdall/src/heimdall/training/training_results.txt"