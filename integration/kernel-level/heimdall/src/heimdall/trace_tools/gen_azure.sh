#!/bin/bash
#original is 77us arrival
#2x 38
#3x is 25us
#4x is 19

# heavy (heavy in io size)
mkdir -p heavy_part2
python3 gen.py heavy_part2/azure1.trace 0.95 5 2/3 8192/9216 208


# light (light in io size)
mkdir -p light_part2
python3 gen.py light_part2/azure1.trace 0.75 5 64/72 11/12 104