import numpy as np
from scipy import stats

def weights(x, t):
  '''Optimal weights for data x and targets t (linear model).'''
  xT = np.transpose(x)
  xTx = np.dot(xT, x)
  xTt = np.dot(xT, t)
  return np.dot(np.linalg.inv(xTx), xTt)

def linear_pred(x, w):
  '''Predict response given data x and weights w, using linear model.'''
  return np.dot(x, w)

def MSE(y, yhat):
  '''Mean squared error for true values y and estimated values yhat.'''
  return np.average(np.square(y - yhat))

def SSR(y, yhat):
  '''Residual sum of squares for true values y and estimated values yhat.'''
  return np.sum(np.square(y - yhat))

def SST(y):
  '''Total sum of squares for true values y and estimated values yhat.'''
  return np.sum(np.square(np.mean(y) - y))

def r_sq(testy, yhat):
  '''Returns the coefficient of determination.'''
  return 1 - SSR(testy, yhat)/SST(testy)

def add_ones(matrix):
  '''Adds a column of ones to the front of the matrix.'''
  return np.append(np.ones((len(matrix),1)), matrix, axis=1)

def my_linear_model((trainx, trainy), (testx, testy)):
  # add column of 1's to X vectors for intercept term
  trainx = add_ones(trainx)
  testx = add_ones(testx)

  w = weights(trainx, trainy)
  yhat = linear_pred(testx, w)
  return MSE(testy, yhat)

def package_linear_model((trainx, trainy), (testx, testy)):
  # add column of 1's to X vectors for intercept term
  trainx = add_ones(trainx)
  testx = add_ones(testx)

  stats.linregress
