# start-up code!
import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from load_cifar10_tvt import load_cifar10_train_val
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10_train_val()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Val data shape: ', X_val.shape
print 'Val labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

if False:
    # Now, implement the vectorized version in softmax_loss_vectorized.

    import time
    from f17cs7643.classifiers.softmax import softmax_loss_vectorized

    W = np.random.randn(10, 3073) * 0.0001

    tic = time.time()
    loss, grad = softmax_loss_vectorized(W, X_train, y_train, 0.00001)
    toc = time.time()
    print 'vectorized loss: %e computed in %fs' % (loss, toc - tic)

    # As a rough sanity check, our loss should be something close to -log(0.1).
    print 'loss: %f' % loss
    print 'sanity check: %f' % (-np.log(0.1))

    # gradient check.
    from f17cs7643.gradient_check import grad_check_sparse
    f = lambda w: softmax_loss_vectorized(w, X_train, y_train, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad, 10)

# Now that efficient implementations to calculate loss function and gradient of the softmax are ready,
# use it to train the classifier on the cifar-10 data
from f17cs7643.classifiers import Softmax
classifier = Softmax()
loss_hist = classifier.train(X_train, y_train, learning_rate=1e-6, reg=1e-6, num_iters=1000,batch_size=128, verbose=True)
# Plot loss vs. iterations
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Evaluate on validation set
y_val_pred = classifier.predict(X_val)
val_accuracy = np.mean(y_val == y_val_pred)
print 'softmax on raw pixels validation set accuracy: %f' % (val_accuracy, )

# Evaluate on test set
y_test_pred = classifier.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'softmax on raw pixels final test set accuracy: %f' % (test_accuracy, )

if True:
    # Visualize the learned weights for each class
    w = classifier.W[:, :-1]  # strip out the bias
    w = w.reshape(10, 32, 32, 3)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in xrange(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

    plt.show()
