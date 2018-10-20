#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 32 \
    --epochs 1 \
    --weight-decay 0.001 \
    --momentum 0.9 \
    --batch-size 50 \
    --lr 0.0001 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
