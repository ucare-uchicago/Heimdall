#!/bin/bash

# check the number of input arguments
if [ $# -lt 4 ]; then
    echo "Incorrect Input format, it should be ./train_linnos.sh device0 device1 dummy_inflection_percentile dir_to_replayed_traces"
    echo "For instance: ./train_linnos.sh nvme0n1 nvme1n1 00 $FLASHNET_INTEGRATION/data/grouping_2_traces_v1.duration_2.5_mins/msr.cut.per_5mins.rw_51_49.774/msr.cut.per_5mins.rw_71_29.565/modified.rerate_2.00...modified.rerate_2.00"
    exit 1
fi
echo "Input $# arguments"

directories=("${@:4:$#}")  # This extracts all arguments except the last two

cd training
idx=1
total_dir=$(( $# - 3 ))
for dir in "${directories[@]}"; do
    # check whether the input directory exists or not.
    if [ ! -d $dir ]; then
        echo "Input directory not exist: $dir"
        exit 1
    fi


    # check whether the baseline is replayed or not
    if [ ! -e "$dir/$1...$2/baseline/trace_1.trace" ] || [ ! -e "$dir/$1...$2/baseline/trace_2.trace" ]; then
        echo "Baseline results: $dir/$1...$2/baseline/ not exist. Please run baseline first."
        exit
    fi

    # echo ""
    # echo "[======================================> Prcessing $idx out of $total_dir <==============================================]"
    echo "Processing Traces in $dir"
    # $3 is the inflection percentile
    ./train_two_ssds.sh $dir $1 $2
    idx=$((idx + 1))

    echo "Done Training"

    if [ -d $dir/$1...$2/linnos ]; then
        echo "Output: $dir/$1...$2/linnos"
    else
        echo "Fail in Training Process."
        exit 1
    fi
done
