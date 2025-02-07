#!/bin/bash

set -e
set -u
set -o pipefail

red="\e[31m"
reset="\e[0m"

FeatureExtractor="feat_v6"
tailAlogrithm="tail_v1"
trainModel="nnK.py"

if [ $# -ne 4 ]
  then
    echo -e "${red}✘ Error: Usage train.sh <trace_0> <trace_1> <device_0> <device_1>${reset}"
    exit 1
fi

# $1: trace_0
# $2: trace_1
# $3: device_0
# $4: device_1
echo $1, $2, $3, $4
mkdir -p mlData

make -C ../io_replayer
sudo ../io_replayer/replayer baseline mlData/TrainTraceOutput 2 "${3}-${4}" $1 $2


for i in 0 1
do
   python3 ./TailAlgorithms/$tailAlogrithm.py -file ./mlData/TrainTraceOutput_baseline.data -output ./mlData/"temp${i}" -device "$i"
done

for i in 0 1
do
    python3 ./FeatureExtractors/$FeatureExtractor.py -files ./mlData/temp${i} -output ./mlData/"mldrive${i}" -device "$i"
done


for i in 0 1
do
   python3 $trainModel \
   -dataset mlData/"mldrive${i}.csv" -train_eval_split 50_50 
   > mlData/"mldrive${i}results".txt
done

cd mlData
mkdir -p drive0weights
mkdir -p drive1weights
cp mldrive0.csv.* drive0weights
cp mldrive1.csv.* drive1weights

cd ..
mkdir -p weights_header_2ssds
python3 mlHeaderGen+2.py Trace dev0 mlData/drive0weights weights_header_2ssds
python3 mlHeaderGen+2.py Trace dev1 mlData/drive1weights weights_header_2ssds