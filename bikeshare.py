import numpy as np
import time

import gp
import linear_regression

# Data field column numbers for explanatory variables
SEASON = 0
HOLIDAY = 1
WORKINGDAY = 2
WEATHER = 3
TEMP = 4
ATEMP = 5
HUMIDITY = 6
WINDSPEED = 7

# My self-defined data field column numbers
MONTH = 8
HOUR = 9
JANUARY = 9
FEBRUARY = 9
MARCH = 9
APRIL = 9
MAY = 9
JUNE = 9
JULY = 9
AUGUST = 9
SEPTEMBER = 9
OCTOBER = 9
NOVEMBER = 9
DECEMBER = 9

# Data field column numbers for response variables
CASUAL = 0
REGISTERED = 1
COUNT = 2


def load_from_file(filename):
  float_data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                             usecols=(1,2,3,4,5,6,7,8,9,10,11))
  explanatory_vars = float_data[:, 0:8]
  response_vars = float_data[:, 8:]

  datetime_data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                                usecols=0, dtype="string")
  explanatory_vars = convert_datetime(explanatory_vars, datetime_data)
  print explanatory_vars[0]
  print explanatory_vars[10885]
  return (explanatory_vars, response_vars)

def convert_datetime(float_data, datetime_data):
  '''Convert datetime string into floats, add columns to existing float_data.'''
  datetime_data = [time.strptime(date, "%Y-%m-%d %H:%M:%S")
                   for date in datetime_data]

  # Only take month and hour
  datetime_data = np.array(datetime_data)[:, 1:4]
  month_col = datetime_data[:, 0]
  hour_col = datetime_data[:, 2]

  # Convert month into 12 binary variables
  months = np.zeros((np.shape(float_data)[0], 12))
  for i, entry in enumerate(month_col):
    months[i, (entry - 1)] = 1

  # Convert hour into 24 binary variables
  hours = np.zeros((np.shape(float_data)[0], 24))
  for i, entry in enumerate(month_col):
    hours[i, (entry - 1)] = 1

  all_data = np.c_[float_data, months, hours]
  return all_data

def scale_explanatory(data):
  '''Scale the explanatory variables to the range [0, 1].'''
  data[:, SEASON] /= 4
  data[:, WEATHER] /= 4
  data[:, TEMP] /= max(data[:, TEMP])
  data[:, ATEMP] /= max(data[:, ATEMP])
  data[:, HUMIDITY] /= 100
  data[:, WINDSPEED] /= max(data[:, WINDSPEED])
  data[:, MONTH] /= 12
  data[:, MDAY] /= 31
  data[:, HOUR] /= 23
  return data

def get_training_data():
  return load_from_file('train.csv')

def get_scaled_training_data():
  explanatory, response = get_training_data()
  return scale_explanatory(explanatory), response

def get_test_data():
  return load_from_file('train.csv')

def get_scaled_test_data():
  explanatory, response = get_test_data()
  return scale_explanatory(explanatory), response

if __name__ == '__main__':
  train = get_training_data()
  test = get_test_data()

  print linear_regression.my_linear_model(train, test)

'''
  # Gaussian process model with linear covariance
  print gp.fit_gaussian_linear_cov(train, test)

  scaled_train = get_scaled_training_data()
  scaled_test = get_scaled_test_data()

  # Gaussian process model with linear covariance and scaled data
  print gp.fit_gaussian_linear_cov(scaled_train, scaled_test)

  # Gaussian process model with hyperparams (concurrent)
  fit_gaussian_nonlinear_cov(train, test)

  # Gaussian process model with hyperparams, rescaled covariates
  gamma_start = 0.1
  gamma_stop = 10
  gamma_step = 0.5
  gamma = (gamma_start, gamma_stop, gamma_step)

  rho_start = 0.01
  rho_stop = 1
  rho_step = 0.05
  rho = (rho_start, rho_stop, rho_step)

  fit_gaussian_nonlinear_cov(train, test, gamma, rho)
'''
