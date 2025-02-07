#!/bin/bash
#1x 416
#2x 208
#4x 104


# light (light in io size)
mkdir -p light_part1
python3 gen.py light_part1/bing_i1.trace 0.75 10 64/72 11/12 104


# heavy (heavy in io size)
mkdir -p heavy_part1
python3 gen.py heavy_part1/bing_i1.trace 0.95 10 2/3 8192/9216 208
