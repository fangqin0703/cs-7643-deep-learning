#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 500 \
    --epochs 10 \
    --weight-decay 1 \
    --momentum 0.9 \
    --batch-size 100 \
    --lr 0.0001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
