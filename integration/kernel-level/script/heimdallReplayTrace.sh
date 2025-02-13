#!/bin/bash

set -e
set -u
set -o pipefail

green="\e[32m"
red="\e[31m"
reset="\e[0m"

SCRIPT_DIR=$(pwd)

# ================================== 1. Validate Configuration =========================================

# Check if the kernel version is '6.0.0-heimdall'
kernel_version=$(uname -r)
if [ "$kernel_version" = "6.0.0-heimdall" ]; then
    echo -e "${green}✔ Kernel version matches 6.0.0-heimdall. ${reset}"
else
    echo -e "${red}✘ Error: The current kernel version is not 6.0.0-heimdall, maybe you should run compileKernelHeimdall.sh first. ${reset}"
    exit 1
fi

# Source the configuration file
source $HEIMDALL_KERNEL/config/config.conf

# format a output directory
ABBREVIATE_SSD_DEVICE0="${SSD_DEVICE0#/dev/}"
ABBREVIATE_SSD_DEVICE1="${SSD_DEVICE1#/dev/}"
ABBREVIATE_TRACE_TO_DDS_DEVICE0=$(echo "$TRACE_TO_SSD_DEVICE0" | rev | cut -d'/' -f1 | cut -d'.' -f2- | rev)
ABBREVIATE_TRACE_TO_DDS_DEVICE1=$(echo "$TRACE_TO_SSD_DEVICE1" | rev | cut -d'/' -f1 | cut -d'.' -f2- | rev)

OUTPUT_DIR="$HEIMDALL_KERNEL/benchmark/results/${ABBREVIATE_TRACE_TO_DDS_DEVICE0}_${ABBREVIATE_TRACE_TO_DDS_DEVICE1}"

if [ ! -d $OUTPUT_DIR ]; then
    echo -e "${red}✘ Error: Training results not exist, please run trainModuleHeimdall.sh first. ${reset}"
    exit 1
fi

# Check whether the config meet each other.
current_config_file=$HEIMDALL_KERNEL/config/config.conf
train_config_file=$OUTPUT_DIR/config_copy.conf
cleaned_config_file1=$(grep -vE '^#|^$' "$current_config_file" | sort)
cleaned_config_file2=$(grep -vE '^#|^$' "$train_config_file" | sort)

# Compare the cleaned files
if diff <(echo "$cleaned_config_file1") <(echo "$cleaned_config_file2") > /dev/null; then
    echo -e "${green}✔ Configuration of Training and Replaying are the same. ${reset}"
else
    echo -e "${red}✘ Error: Training and Replaying use different configuration. ${reset}"
    exit 1
fi

# Check whether trained weights exist
if [ ! -f "$OUTPUT_DIR/heimdall/w_Trace_dev0.h" ]; then
    echo -e "${red}✘ Error: Trained weight: w_Trace_$ABBREVIATE_SSD_DEVICE0.h not exist. Please run trainModuleHeimdall.sh first. ${reset}"
    exit 1
fi
if [ ! -f "$OUTPUT_DIR/heimdall/w_Trace_dev1.h" ]; then
    echo -e "${red}✘ Error: Trained weight: w_Trace_$ABBREVIATE_SSD_DEVICE1.h not exist. Please run trainModuleHeimdall.sh first. ${reset}"
    exit 1
fi
echo -e "${green}✔ Find header files of trained weights. ${reset}"


# ===================================== 2. Make ============================================
# make necessary modules
cd $HEIMDALL_KERNEL/heimdall/src/kapi/uspace
make clean 
make

cd $HEIMDALL_KERNEL/heimdall/src/kapi/kshm
make clean 
make

cd $HEIMDALL_KERNEL/heimdall/src/kapi/kernel
make clean 
make

cd $SCRIPT_DIR
echo -e "${green}✔ Make kapi done. ${reset}"

# make io_replayer
cd $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer
sudo make
cd $SCRIPT_DIR
echo -e "${green}✔ Make io_replayer done. ${reset}"

# copy the weights to heimdall module
rm -rf $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook/weights_header/*
if [ ! -d $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook/weights_header ]; then
    mkdir -p $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook/weights_header
fi
cp $OUTPUT_DIR/heimdall/w_Trace_dev0.h $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook/weights_header/
cp $OUTPUT_DIR/heimdall/w_Trace_dev1.h $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook/weights_header/

C_FILE="$HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook/main.c"

# Use awk to replace the second occurrence of "/dev/xxx" with SSD_DEVICE0
awk -v newDevice="${SSD_DEVICE0}" '
    /"\/dev\/[^"]*"/ {
        count++
        if (count == 1) {
            sub(/"\/dev\/[^"]*"/, "\"" newDevice "\"")
        }
    }
    { print }
' "$C_FILE" > temp_file && mv temp_file "$C_FILE"

# Use awk to replace the second occurrence of "/dev/xxx" with SSD_DEVICE1
awk -v newDevice="${SSD_DEVICE1}" '
    /"\/dev\/[^"]*"/ {
        count++
        if (count == 2) {
            sub(/"\/dev\/[^"]*"/, "\"" newDevice "\"")
        }
    }
    { print }
' "$C_FILE" > temp_file && mv temp_file "$C_FILE"

# make heimdall
cd $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook
make
echo -e "${green}✔ Make kernel hook done. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall
make
echo -e "${green}✔ Make heimdall done. ${reset}"


# ===================================== 3. Replay traces ============================================

# 0. Warmup heimdall (FixMe in the future, this is not a good way to make it general works on machines)
cd $HEIMDALL_KERNEL/heimdall/src/kapi

sudo ./load.sh > load.log 2>&1 &
PID=$!

# Give the process some time to start up and log initial messages
sleep 4

if grep -q "Error" load.log; then
    echo -e "${red}✘ Error: detected startup error in load.sh ${reset}"
    kill $PID
    exit 1
else
    rm -rf load.log
fi
echo -e "${green}✔ Modules Inserted. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook
./enable_heimdall.sh
echo -e "${green}✔ Enable heimdall. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer
echo "Replaying traces with the help of heimdall..."
./run_2ssds.sh heimdall $TRACE_TO_SSD_DEVICE0 $TRACE_TO_SSD_DEVICE1 $SSD_DEVICE0 $SSD_DEVICE1 1
echo -e "${green}✔ Replaying traces with heimdall done. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook
./disable_heimdall.sh
echo -e "${green}✔ Disable heimdall. ${reset}"

# 1. Run heimdall
cd $HEIMDALL_KERNEL/heimdall/src/kapi

sudo ./load.sh > load.log 2>&1 &
PID=$!

# Give the process some time to start up and log initial messages
sleep 4

if grep -q "Error" load.log; then
    echo -e "${red}✘ Error: detected startup error in load.sh ${reset}"
    kill $PID
    exit 1
else
    rm -rf load.log
fi
echo -e "${green}✔ Modules Inserted. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook
./enable_heimdall.sh
echo -e "${green}✔ Enable heimdall. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer
echo "Replaying traces with the help of heimdall..."
./run_2ssds.sh heimdall $TRACE_TO_SSD_DEVICE0 $TRACE_TO_SSD_DEVICE1 $SSD_DEVICE0 $SSD_DEVICE1 0
echo -e "${green}✔ Replaying traces with heimdall done. ${reset}"

cd $HEIMDALL_KERNEL/heimdall/src/heimdall/kernel_hook
./disable_heimdall.sh
echo -e "${green}✔ Disable heimdall. ${reset}"

# 2. Run baseline
cd $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer
echo "Replaying traces (baseline)..."
./run_2ssds.sh baseline $TRACE_TO_SSD_DEVICE0 $TRACE_TO_SSD_DEVICE1 $SSD_DEVICE0 $SSD_DEVICE1 0
echo -e "${green}✔ Replaying traces(baseline) done. ${reset}"

# 3. Run random
cd $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer
echo "Replaying traces (random)..."
./run_2ssds.sh random $TRACE_TO_SSD_DEVICE0 $TRACE_TO_SSD_DEVICE1 $SSD_DEVICE0 $SSD_DEVICE1 0
echo -e "${green}✔ Replaying traces(random) done. ${reset}"

# # store the replaying results
cp $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer/2ssds_heimdall.data $OUTPUT_DIR/heimdall/
echo "heimdall's replayed result stored in: $OUTPUT_DIR/heimdall/2ssds_heimdall.data"

mkdir -p $OUTPUT_DIR/baseline
cp $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer/2ssds_baseline.data $OUTPUT_DIR/baseline/
echo "baseline's replayed result stored in: $OUTPUT_DIR/baseline/2ssds_baseline.data"

mkdir -p $OUTPUT_DIR/random
cp $HEIMDALL_KERNEL/heimdall/src/heimdall/io_replayer/2ssds_random.data $OUTPUT_DIR/random/
echo "random's replayed result stored in: $OUTPUT_DIR/random/2ssds_random.data"

echo -e "${green}✔ Done. ${reset}"