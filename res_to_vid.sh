#!/bin/bash

if [ $1 = 'all' ]; then
    for clip in $(ls result); do
        for model in $(ls result/$clip); do
            for mode in $(ls result/$clip/$model); do
                python encoding.py result/$clip/$model/$mode
            done
        done
    done
elif [ -d $1 ]; then
    python encoding.py $1
else
    echo Usage: bash allencoding.sh "{{all, dir}}"
fi