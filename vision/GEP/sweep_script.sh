#!/bin/bash

echo "Running perp lr 0.01..."
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --perp --lr 0.01 --n_epoch 25

echo "Running no perp lr 0.01..."
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --lr 0.01 --n_epoch 25

echo "Running perp lr 0.1 clip0 0.1"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --perp --lr 0.1 --n_epoch 25 --clip0 0.1

echo "Running no perp lr 0.1 clip0 0.1"
echo "Start time: $(date +'%T')"
python3 ./vision/GEP/main.py -p --lr 0.1 --n_epoch 25 --clip0 0.1
echo "End time: $(date +'%T')"
