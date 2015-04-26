import numpy as np
import time

# Data field column numbers
SEASON = 0
HOLIDAY = 1
WORKINGDAY = 2
WEATHER = 3
TEMP = 4
ATEMP = 5
HUMIDITY = 6
WINDSPEED = 7
CASUAL = 8
REGISTERED = 9
COUNT = 10

# My self-defined data field column numbers
YEAR = 11
MONTH = 12
MDAY = 13
HOUR = 14


def load_from_file(filename):
  float_data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                             usecols=(1,2,3,4,5,6,7,8,9,10,11))
  datetime_data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                                usecols=0, dtype="string")
  data = convert_datetime(float_data, datetime_data)
  return data

def convert_datetime(float_data, datetime_data):
  '''Insert custom data fields.'''
  datetime_data = [time.strptime(date, "%Y-%m-%d %H:%M:%S") for date in datetime_data]
  all_data = np.c_[float_data, datetime_data]
  return all_data[:, 0:15] # Ignore datetime info after hour

def scale(data):
  '''Scale the data to the range [0, 1].'''
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

if __name__ == '__main__':
  data = load_from_file('train.csv')
  data = insert_custom_fields(data)
  data = scale(data)
