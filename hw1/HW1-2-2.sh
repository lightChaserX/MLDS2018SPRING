#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python HW1-2-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 0 --batch-size 128 &
CUDA_VISIBLE_DEVICES=1 python HW1-2-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 0 --batch-size 128 &
CUDA_VISIBLE_DEVICES=1 python HW1-2-2.py --num-model 2 --lr 0.001 --epochs 1000 --times 0 --batch-size 128 &
