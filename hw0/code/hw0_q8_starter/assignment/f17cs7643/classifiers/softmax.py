import numpy as np
from random import shuffle


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Loss
    N = X.shape[1]
    scores = np.matmul(W, X)
    scores -= np.max(scores, axis=0)    # for numerical stability so that we don't divide large numbers
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)
    log_probs = np.log(probs[y, range(N)])    # use numpy multidimensional array indexing
    loss = -np.sum(log_probs) / N + reg*np.sum(W*W)

    # Gradient
    grad = probs
    grad[y, range(N)] -= 1
    dW = 1.0/N*np.matmul(grad, X.T) + 2*reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
