#!/bin/bash

rm -rf trace*

mkdir traceA
cat light_part1/bing_i1.trace > traceA/traceA.trace
cat heavy_part2/azure1.trace >> traceA/traceA.trace 

mkdir traceB
cat heavy_part1/bing_i1.trace > traceB/traceB.trace
cat light_part2/azure1.trace >> traceB/traceB.trace 

cp -r traceA /mnt/prefetcher_exp/Heimdall/integration/kernel-level/benchmark/azure_bing_i/
cp -r traceB /mnt/prefetcher_exp/Heimdall/integration/kernel-level/benchmark/azure_bing_i/

rm -rf light_part*
rm -rf heavy_part*