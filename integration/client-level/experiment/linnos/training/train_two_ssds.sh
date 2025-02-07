#!/bin/bash

TraceTag='trace'

if [ $# -ne 3 ]
  then
    echo "Usage train.sh <trace_dir> <device0> <device1>"
    # eg : ./train.sh testTraces/hacktest.trace testTraces/hacktest.trace testTraces/hacktest.trace
    exit
fi

# get the replayed results of baseline
echo "Trace Dir => $1"
echo "Device 0 => $2"
echo "Device 1 => $3"

# create a place to store the intermediate and final results of training
training_result_dir="$1/$2...$3/linnos/training_results"
mkdir -p $training_result_dir

echo $(pwd)
for i in 0 1     # i here represent device index
do
   python3 ./traceParser.py direct 3 4 \
   "$1/$2...$3/baseline/trace_$((i+1)).trace" $training_result_dir/"temp${i}" \
   $training_result_dir/"mldrive${i}.csv" "$i"
done

for i in 0 1
do
   ip_percentile=`./calc_inflection_point.py -input_file "$1/$2...$3/baseline/trace_$((i+1)).trace"`
   echo "============= Inflection Point (IP) Percentile for device $i is $ip_percentile"
   python3 ./pred1.py \
   $training_result_dir/"mldrive${i}.csv" $ip_percentile > $training_result_dir/"mldrive${i}results".txt
done

mkdir -p $training_result_dir/drive0weights
mkdir -p $training_result_dir/drive1weights
cp $training_result_dir/mldrive0.csv.* $training_result_dir/drive0weights
cp $training_result_dir/mldrive1.csv.* $training_result_dir/drive1weights

# generate the header file
mkdir -p $training_result_dir/weights_header_2ssds
python3 ./mlHeaderGen.py Trace "dev_0" $training_result_dir/drive0weights $training_result_dir/weights_header_2ssds
python3 ./mlHeaderGen.py Trace "dev_1" $training_result_dir/drive1weights $training_result_dir/weights_header_2ssds