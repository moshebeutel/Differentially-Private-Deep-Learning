#!/bin/bash

echo "Running perp lr 0.01 clip0 0.01"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --perp --lr 0.01 --n_epoch 25 --clip0 0.01

echo "Running no perp lr 0.01 clip0 0.01"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --lr 0.01 --n_epoch 25 --clip0 0.01

echo "Running perp lr 0.1 clip0 0.01"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --perp --lr 0.1 --n_epoch 25 --clip0 0.01

echo "Running no perp lr 0.1 clip0 0.01"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --lr 0.1 --n_epoch 25 --clip0 0.01
echo "End time: $(date +'%T')"
