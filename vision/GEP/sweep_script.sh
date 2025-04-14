#!/bin/bash

#echo "Running perp lr 0.01 clip0 0.01"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --perp --lr 0.01 --n_epoch 25 --clip0 5.0

#echo "Running no perp lr 0.01 clip0 0.01"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --lr 0.01 --n_epoch 25 --clip0 5.0
#
#echo "Running perp lr 0.1 clip0 0.01"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --perp --lr 0.1 --n_epoch 25 --clip0 5.0
#
#echo "Running no perp lr 0.1 clip0 0.01"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --lr 0.1 --n_epoch 25 --clip0 5.0
#echo "End time: $(date +'%T')"


#
#echo "Running no perp lr 0.01 clip0 5.0"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --lr 0.01 --n_epoch 5 --clip0 5.0
#
echo "Running no perp lr 0.1 clip0 5.0"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --lr 0.1 --n_epoch 25 --clip0 5.0
#
#echo "Running no perp lr 1.0 clip0 5.0"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --lr 1.0 --n_epoch 5 --clip0 5.0

#echo "Running no perp lr 0.1 clip0 1.0"
#echo "Start time: $(date +'%T')"
#python3 ./vision/GEP/main.py -p --lr 0.1 --n_epoch 5 --clip0 1.0