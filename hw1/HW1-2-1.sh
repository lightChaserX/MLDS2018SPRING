#!/bin/sh
model_max=7
for i in `seq 0 $model_max`
do
        echo Model $i
        CUDA_VISIBLE_DEVICES=$1 python HW1-2-1.py --num-model $2 --num-func $3 --lr 0.0001 --momentum 0 --iterations 20000 --times $i
done

