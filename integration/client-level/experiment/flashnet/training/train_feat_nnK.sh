#!/bin/bash
FeatureExtractor="feat_v6"
tailAlogrithm="tail_v1"
trainModel="nnK.py"

# get the replayed results of baseline
echo "Trace Dir => $1"
echo "Device 0 => $2"
echo "Device 1 => $3"

# create a place to store the intermediate and final results of training
training_result_dir="$1/$2...$3/flashnet/training_results"
mkdir -p $training_result_dir


for i in 0 1
do
   python3 ./TailAlgorithms/$tailAlogrithm.py -file "$1/$2...$3/baseline/trace_$((i+1)).trace" -output $training_result_dir/"temp${i}"
done

for i in 0 1
do
    python3 ./FeatureExtractors/$FeatureExtractor.py -files $training_result_dir/temp${i} -output $training_result_dir/"mldrive${i}" -device "$i"
done

echo "Done Feature Extraction"

for i in 0 1
do
   # mldrive*.csv is the same as profile_v1.feat_v6.*.dataset
   
   # Mode 1: Using read and write IO as training data
   python3 $trainModel -dataset $training_result_dir/"mldrive${i}.csv" -train_eval_split 50_50 > $training_result_dir/"mldrive${i}results".txt
   
   # Mode 2: Using read ONLY IO as training data
   # python3 $trainModel -dataset $training_result_dir/"mldrive${i}.readonly.csv" -train_eval_split 50_50 > $training_result_dir/"mldrive${i}results".txt
done

echo "Done Training"

mkdir -p $training_result_dir/drive0weights
mkdir -p $training_result_dir/drive1weights
cp $training_result_dir/mldrive0.csv.* $training_result_dir/drive0weights
cp $training_result_dir/mldrive1.csv.* $training_result_dir/drive1weights

echo "Done copying weights"

# generate the header file
mkdir -p $training_result_dir/weights_header_2ssds
python3 mlHeaderGen+2.py Trace dev_0 $training_result_dir/drive0weights $training_result_dir/weights_header_2ssds 
python3 mlHeaderGen+2.py Trace dev_1 $training_result_dir/drive1weights $training_result_dir/weights_header_2ssds 

echo "Done generating header files"