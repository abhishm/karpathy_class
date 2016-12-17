import numpy as np
from random import shuffle

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
  num_train, D = X.shape
  num_classes = np.max(y)+1
  for i in xrange(num_train):
    cwg = np.dot(X[i, :], W)
    cwg -= np.max(cwg)
    loss += -cwg[y[i]]+np.log(np.sum(np.exp(cwg)))
    
    for j in xrange(num_classes):
        if j != y[i]:
            dW[:,j] += (np.exp(cwg[j])/np.sum(np.exp(cwg)))*X[i]
        else: 
            dW[:,j] += (np.exp(cwg[j])/np.sum(np.exp(cwg)))*X[i] - X[i]
  
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
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
  num_train, D = X.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  cwg = np.dot(X,W)
  cwg -= np.max(cwg, axis=1).reshape(-1,1)
  cwg_exp = np.exp(cwg)
  loss += np.sum(-np.log(cwg_exp[np.arange(num_train), y]/cwg_exp.sum(axis=1)))
  
  cwg_exp_dW = (cwg_exp/np.sum(cwg_exp, axis=1).reshape(-1,1))
  dZ = cwg_exp_dW 
  dZ[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, dZ)
  #for i in xrange(num_train):
  #  dW += np.outer(cwg_exp_dW[i], X[i]).T
  #  dW[:,y[i]] -= X[i]
    
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

