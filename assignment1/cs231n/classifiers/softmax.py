import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    pro = np.dot(X[i], W)
    pro -= np.max( pro)

    pro = np.exp(pro)

    pro /= np.sum(pro)

    loss += -np.log(pro[y[i]])

    one_hot = np.zeros(num_classes)
    one_hot[y[i]] = 1
    error = pro - one_hot
    dW += np.outer( X[i], error)

  loss /= num_train
  loss +=  0.5 * reg *  np.sum(W*W)
  dW /= num_train

  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  pro = np.dot(X,W)
  pro -= np.max(pro, axis = 1 ).reshape(-1, 1)

  pro = np.exp(pro)
  pro /= np.sum(pro, axis= 1 ).reshape(-1,1)

  true_y = np.zeros([X.shape[0], W.shape[1]])
  true_y[range(X.shape[0]), y] = 1
  loss =  np.multiply(np.log(pro), true_y) 

  loss = - np.sum( loss )/X.shape[0]
  loss += 0.5 * reg * np.sum(W*W)

  error = pro - true_y

  dW = np.dot( X.T, error )

  dW /=X.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

