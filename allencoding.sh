#!/bin/bash

for model in $(ls result); do
    for mode in $(ls result/$model); do
        for clip in $(ls result/$model/$mode); do
            python encoding.py result/$model/$mode/$clip
        done
    done
done