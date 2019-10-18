#!/bin/bash

for i in `seq 1 23`;
do
    echo "------------------------------------"
    echo "Generating stimuli for problem $i..."
    python generate.py \
        --problem "$i" \
        --nb_samples 100 \
        --data_dir data/images \
        --parsed_dir data/parsed --parsed_dir_classic data/parsed_classic \
        --symb_h5_raw_dir data/symb_raw --symb_h5_obf_dir data/symb_obf
done
