#!/bin/bash
python main.py        \
    --train_newgnn    \
    --train_defectnode \
    --generate_new_sample \
    --generate_sample_badedge \
    --numofepoch 2000 \
    --subblock_size 10 \
    --k1  100  \
    --k2  10  \
    --inilr    0.0001




