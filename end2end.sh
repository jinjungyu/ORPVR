#!/bin/bash
while read clip; do 
    # python masking.py rawdataset/DAVIS_640x480/$clip
    for model in aotgan e2fgvi_hq; do
        # python inpainting.py dataset/$clip --model $model
        for mode in 0 1 2; do
            python relocating.py result_inpaint/$model/$clip --mode $mode
        done
    done

done < data_experiment.txt