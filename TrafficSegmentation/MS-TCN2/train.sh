#!/bin/bash

python main.py --action=train --dataset=${1} --split=${2} \
                --num_epochs=1000 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --num_f_maps=64 \
                --lr=0.0001 \
                --bce_pos_weight=1 \
                --bz=8

