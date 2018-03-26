#!/bin/sh
model_max=2
for i in `seq 0 $model_max`
do
	echo Model $i
	python HW1-1-1.py --num-model $i --iterations 10000
done
