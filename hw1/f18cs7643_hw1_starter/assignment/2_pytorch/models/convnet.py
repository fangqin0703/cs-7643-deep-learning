import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        channels, height, width = im_size
        self.filter_size = kernel_size
        self.num_filters = hidden_dim
        self.conv_param = {'stride': 1, 'pad': (self.filter_size - 1) // 2}
        self.pool_size = 2
        self.conv = nn.Conv2d(channels, self.num_filters, kernel_size=self.filter_size,
                              stride=self.conv_param['stride'], padding=self.conv_param['pad'])
        self.activations = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=self.pool_size)
        self.affine = nn.Linear(self.num_filters * (height // self.pool_size) * (width // self.pool_size), n_classes)
        self.softmax = nn.Softmax()
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        N = images.shape[0]
        output = self.conv(images)
        output = self.activations(output)
        output = self.pool(output)
        output = self.affine(output.view(N, -1))
        #scores = self.softmax(output)
        scores = output
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

