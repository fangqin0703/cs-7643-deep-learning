import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_mat = np.reshape(x, (x.shape[0], -1))
  out = np.matmul(x_mat, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  x = np.reshape(x, (x.shape[0], -1))
  dw = np.matmul(x.T, dout)
  db = np.sum(dout, axis=0)
  dx = np.matmul(dout, w.T)
  dx = np.reshape(dx, cache[0].shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride, pad_width = conv_param['stride'], conv_param['pad']
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')   # default padding value is zero
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  h_out = np.int(1 + (H + 2 * pad_width - HH) / stride)
  w_out = np.int(1 + (W + 2 * pad_width - WW) / stride)
  out = np.zeros((N, F, h_out, w_out), dtype=x.dtype)

  for n in range(N):
    for f in range(F):
      for row in range(0, H + 2 * pad_width - HH + 1, stride):
        for col in range(0, W + 2 * pad_width - WW + 1, stride):
          tmp = x_padded[n, :, row : row + HH, col : col + WW]
          out[n, f, np.int(row / stride), np.int(col / stride)] = \
            np.sum(np.multiply(x_padded[n, :, row : row + HH, col : col + WW], w[f, :, :, :])) + b[f]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  stride, pad_width = conv_param['stride'], conv_param['pad']
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)),
                    mode='constant')  # default padding value is zero
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  h_out = np.int(1 + (H + 2 * pad_width - HH) / stride)
  w_out = np.int(1 + (W + 2 * pad_width - WW) / stride)
  N, F, h_out, w_out = dout.shape
  dx = np.zeros(x.shape)
  dx_padded = np.zeros(x_padded.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # dw: convolve x with dout
  if False:
    for f in range(F):
      for c in range(C):
        for row in range(0, H + 2 * pad_width - h_out + 1, stride):
          for col in range(0, W + 2 * pad_width - w_out + 1, stride):
            dw[f, c, row, col] = np.sum(np.multiply(x_padded[:, c, row : row + h_out, col : col + w_out], dout[:, f, :, :]))

  w_flipped = np.flip(w, 2)
  w_flipped = np.flip(w_flipped, 3)
  for f in range(F):
      for row in range(h_out):
        for col in range(w_out):
          for n in range(N):
            #dw[f, c, row, col] = np.sum(np.multiply(x_padded[:, c, row : row + h_out, col : col + w_out], dout[:, f, :, :]))
            dw[f, :, :, :] += x_padded[n, :, row * stride: row * stride + HH, col * stride: col * stride + WW] * dout[n, f, row, col]
            dx_padded[n, :, row * stride : row * stride + HH, col * stride : col * stride + WW] += \
              w[f, :, :, :] * dout[n, f, row, col]
  dx = dx_padded[:, :, pad_width : -pad_width, pad_width : -pad_width]
  if False:
    # dx: convolve dout with flipped w
    dout_padded = np.pad(dout, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)),
                      mode='constant')  # default padding value is zero
    w_flipped = np.flip(w, 2)
    w_flipped = np.flip(w_flipped, 3)
    for n in range(N):
      for c in range(C):
        for row in range(0, h_out + 2 * pad_width - HH + 1, stride):
          for col in range(0, w_out + 2 * pad_width - WW + 1, stride):
            #dx[n, c, row, col] = np.sum(np.multiply(dout_padded[n, :, row : row + HH, col : col + WW], w[:, c, :, :]))
            dx[n, c, row, col] = \
              np.sum(np.multiply(dout_padded[n, :, row: row + HH, col: col + WW], w_flipped[:, c, :, :]))

  # db: sum over dout
  db = np.sum(dout, axis=(0, 2, 3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  out = np.zeros((N, C, np.int(H / stride), np.int(W / stride)), dtype=x.dtype)
  for n in range(N):
    for c in range(C):
      for row in range(0, H - pool_height + 1, stride):
        for col in range(0, W - pool_width + 1, stride):
          out[n, c, np.int(row / stride), np.int(col / stride)] = \
            np.max(x[n, c, row : row + pool_height, col : col + pool_width])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  dx = np.zeros(x.shape, dtype=x.dtype)
  N, C, H, W = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  for n in range(N):
    for c in range(C):
      for row in range(0, H - pool_height + 1, stride):
        for col in range(0, W - pool_width + 1, stride):
          idx = np.argmax(x[n, c, row : row + pool_height, col : col + pool_width])
          dx[n, c, row + np.floor_divide(idx, pool_width), col + np.remainder(idx, pool_width)] = \
            dout[n, c, np.int(row / stride), np.int(col / stride)]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

