#!/bin/bash

python main.py --action=train \
				--arch="asformer"
                --num_epochs=400 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --num_f_maps=64 \
                --lr=0.0005 \
                --bce_pos_weight=1 \
                --bz=1

