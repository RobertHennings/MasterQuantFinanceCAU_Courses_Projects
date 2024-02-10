# C-Exercise 08, SS 2023

import math
import numpy as np

def MC_integration(N, f, a, b):
    # generate N samples in the interval [a,b}
    x_samples = np.random.uniform(a,b, N)
    # compute the corresponding values f(x)
    y_values = f(x_samples)
    # sum them up and multiply by (b-a)/N
    integral = (b-a) * np.sum(y_values)/N
    return integral

def f(x):
    return np.sqrt(1-np.power(x,2))

a = 0
b = 1
for N in [10,100,1000,10000]:
    value = MC_integration(N, f, a, b)
    print("The value of the integral with N=" + str(N) + " is: " + str(value))