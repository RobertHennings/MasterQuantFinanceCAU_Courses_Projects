# C-Exercise 04, SS 2023

import math
import re
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# part a)
# computes the price of european calls in the CRR model
def CRR_EuCall(S_0, r, sigma, T, M, K):
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    q = (math.exp(r * delta_t) - d) / (u - d)

    # allocate matrix S
    S = np.empty((M + 1, M + 1))

    # fill matrix S with stock prices
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)

    # V will contain the call prices
    V = np.empty((M + 1, M + 1))
    # compute the prices of the call at time T
    V[:, M] = np.maximum(S[:, M] - K, 0)

    # define recursion function
    def g(k):
        return math.exp(-r * delta_t) * (q * V[1:k + 1, k] +
                                         (1 - q) * V[0:k, k])

    # compute call prices at t_i
    for k in range(M, 0, -1):
        V[0:k, k - 1] = g(k)

    # return the price of the call at time t_0 = 0
    return V[0, 0]


# part b)
# computes the price of a call in the Black-Scholes model
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) *
           (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C


# part c)
# test parameters
S_0 = 100
r = 0.03
sigma = 0.3
T = 1
K = range(70, 201, 1)
M = 100

V_0 = np.empty(131, dtype=float)
V_0_BS = np.empty(131, dtype=float)

for i in range(0, len(K)):
    V_0[i] = CRR_EuCall(S_0, r, sigma, T, M, K[i])
    V_0_BS[i] = BlackScholes_EuCall(0, S_0, r, sigma, T, K[i])

# plot the error of the approximation against the real Black-Scholes price
plt.clf()
plt.plot(K, V_0_BS - V_0, 'red', label='Error for original conditions')
plt.xlabel('Strike price')
plt.ylabel('Deviation from real BS-price')
plt.legend()

# plot the absolute error in a second window
plt.figure()
plt.clf()
plt.plot(K,
         abs(V_0_BS - V_0),
         'red',
         label='Absolute error for original conditions')
plt.xlabel('Strike price')
plt.ylabel('Deviation from real BS-price')
plt.legend()
plt.show()