#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 0 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 1 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 2 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 3 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 4 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 5 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 6 &
CUDA_VISIBLE_DEVICES=0 python HW1-1-2.py --num-model 0 --lr 0.001 --epochs 1000 --times 7 &

CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 0 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 1 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 2 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 3 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 4 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 5 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 6 &
CUDA_VISIBLE_DEVICES=1 python HW1-1-2.py --num-model 1 --lr 0.001 --epochs 1000 --times 7 &