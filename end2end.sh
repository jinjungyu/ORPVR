#!/bin/bash

modes=("original" "offset" "dynamic")

if [ $1 = 'all' ]; then
    while read clip; do
        python masking.py rawdataset/DAVIS_640x480/$clip
        for model in aotgan e2fgvi_hq; do
            python inpainting.py dataset/$clip --model $model
            for mode in 0 1 2; do
                python relocating.py result_inpaint/$clip/$model --mode $mode
                # python encoding.py result/$clip/$model/modes[$mode]
            done
        done
    done < data_experiment.txt
elif [ -d $1 ]; then
    if [ $2 = 'all' ]; then
        clip=$(basename $1)
        python masking.py $1
        for model in aotgan e2fgvi_hq; do
            python inpainting.py dataset/$clip --model $model
            for mode in 0 1 2; do
                python relocating.py result_inpaint/$clip/$model --mode $mode
                # python encoding.py result/$clip/$model/modes[$mode]
            done
        done
    elif [ $# -eq 3 ]; then
        clip=$(basename $1)
        python masking.py $1
        python inpainting.py dataset/$clip --model $2
        python relocating.py result_inpaint/$clip/$2 --mode $3
        # python encoding.py result/$clip/$2/modes[$3]
    else
        echo "Usage for single clip : bash end2end.sh clip model mode"
    fi
else
    echo -e "Usage: 1. bash end2end.sh all (require data_experiment.txt)
       2. bash end2end.sh clippath all
       3. bash end2end.sh clippath model mode"
fi