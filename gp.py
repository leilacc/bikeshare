import math
import numpy as np
import concurrent.futures

def linearK(x1, x2, **hyperparams):
  '''Noise-free linear cov function for Gaussian process model.'''
  return (100**2)*np.dot(x1, x2)

def nonlinearK(x1, x2, gamma, rho):
  '''Noise-free linear cov function for Gaussian process model.'''
  return (100**2) + (gamma**2)*math.exp(-(rho**2)*np.sum(np.square(x1 - x2)))

def C(x1, x2, K, Binv, gamma=None, rho=None):
  '''Covariance function for Gaussian process model.'''
  return K(x1, x2, gamma=gamma, rho=rho)

def C_N(X, K, Binv, gamma=None, rho=None):
  '''Returns the NxN covariance matrix C_N for data X, kernel K, and beta B.'''
  len_x = len(X)
  matrix = np.zeros(len_x**2).reshape(len_x, len_x)
  for i, x1 in enumerate(X):
    for j, x2 in enumerate(X):
      matrix[i, j] = C(x1, x2, K, Binv, gamma, rho)
      if i == j:
        matrix[i, j] += 1
  return matrix

def k(X, new_x, kernel, gamma=None, rho=None):
  '''Calculates kernel function for all x1..xN in X, with new_x.'''
  vec = np.zeros(len(X))
  for i, x in enumerate(X):
    vec[i] = kernel(x, new_x, gamma=gamma, rho=rho)
  return vec

def m(new_x, X, t, kernel, C_Ninv, gamma=None, rho=None):
  '''Returns mean of predictive dist for new_x given x1 ... xN as X, and t.'''
  kT = np.transpose(k(X, new_x, kernel, gamma, rho))
  return np.dot(np.dot(kT, C_Ninv), t)

def gaussian_pred(trainX, trainY, testX, K, gamma=None, rho=None):
  '''Predict responses testY for testX given a Gaussian Process model specified
  by trainX, trainY, kernel function K and optional hyperparams.'''
  predictions = np.zeros(len(testX))

  B = 1
  C_Ninv = np.linalg.inv(C_N(trainX, K, B**(-1), gamma, rho))

  for i, x in enumerate(testX):
    mean = m(x, trainX, trainY, K, C_Ninv, gamma=gamma, rho=rho)
    predictions[i] = mean
  return predictions

def cross_validate_params(X, Y, params):
  set_size = len(X)/10
  gamma = params[0]
  rho = params[1]

  mses = np.zeros(10)
  for i in range(10):
    lower = i*set_size
    upper = i*set_size + set_size
    testx = X[lower:upper]
    testy = Y[lower:upper]
    trainx = np.vstack([X[0:lower], X[upper:]])
    trainy = np.concatenate((Y[0:lower], Y[upper:]))

    pred = gaussian_pred(trainx, trainy, testx, nonlinearK, gamma, rho)
    mses[i] = MSE(testy, pred)

  return (np.average(mses), gamma, rho)

def cross_validate_concurrent(X, Y, gamma_start, gamma_stop, gamma_step,
                              rho_start, rho_stop, rho_step):
  '''10-fold cross validation on data X of length 250.'''
  set_size = len(X)/10
  best_hyperparams = (0, 0)
  best_mse = float('inf')
  hyperparams = []

  for gamma in np.arange(gamma_start, gamma_stop, gamma_step):
    for rho in np.arange(rho_start, rho_stop, rho_step):
      hyperparams.append((gamma, rho))

  with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
    results = executor.map(cross_validate_params, [X]*400, [Y]*400, hyperparams)
    return list(results)

def fit_gaussian_linear_cov((trainx, trainy), (testx, testy)):
  predictions = gaussian_pred(trainx, trainy, testx, linearK)
  return MSE(testy, predictions)

def fit_gaussian_nonlinear_cov((trainx, trainy), (testx, testy),
                               (gamma_start, gamma_stop, gamma_step),
                               (rho_start, rho_stop, rho_step)):
  results = cross_validate_concurrent(trainx, trainy, gamma_start, gamma_stop,
                                      gamma_step, rho_start, rho_stop, rho_step)
  sorted_results = sorted(results, key=lambda tup: tup[0])
  gamma = sorted_results[0][1]
  rho = sorted_results[0][2]
  print('best params - gamma: %s rho: %s' % (gamma, rho))
  predictions = gaussian_pred(trainx, trainy, testx, nonlinearK, gamma, rho)
  return MSE(testy, predictions)
