# C-Exercise 03, SS 2023

import math
import numpy as np
import matplotlib.pyplot as plt


# part a)
# compute log-returns
def log_returns(data):
    # np.log applies the logarithm to each element of data
    # np.diff(x1, x2, ..., xn) returns (x2-x1, x3-x2, ... , xn-xn-1)
    return np.diff(np.log(data))


# part b)
dax = np.genfromtxt('time_series_dax_2023.csv', delimiter=';', usecols=4, skip_header=1)
dax = np.flip(dax)

# remove NaN, remember ~ is the logical-not operator and np.isnan returns a boolean array
dax = dax[~np.isnan(dax)]

# apply function log_returns to dax-data
lr = log_returns(dax)

# compute mean and standard deviation (=root of the variance) of log-returns
ev = np.mean(lr)
std_dev = math.sqrt(np.var(lr, ddof=1))
print('DAX log-returns: annualized mean = ' + str(ev * 250) + ', annualized standard deviation = ' + str(
    std_dev * math.sqrt(250)) + '.')

# part c)
# simulate log-returns
lr_simulated = np.random.normal(ev, std_dev, len(lr))

# plot for part b) and c)
# plot the log-returns
plt.clf()
plt.plot(lr, 'b')
plt.plot(lr_simulated, 'r')
plt.title('log-returns of DAX in the period 01.01.1990 - 08.04.2022')
plt.xlabel('trading day')
plt.ylabel('log-return')

plt.show()
