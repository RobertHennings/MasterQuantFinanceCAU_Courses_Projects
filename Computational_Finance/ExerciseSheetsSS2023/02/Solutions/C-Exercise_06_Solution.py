# C-Exercise 06, SS 2023

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# part a)
# computes the price of american and european puts in the CRR model
def CRR_AmEuPut(S_0, r, sigma, T, M, K, EU):
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

    # V will contain the put prices
    V = np.empty((M + 1, M + 1))
    # compute the prices of the put at time T
    V[:, M] = np.maximum(K - S[:, M], 0)

    # define recursion function for european and american options
    if (EU == 1):
        def g(k):
            # martingale property in the european case
            return math.exp(-r * delta_t) * (q * V[1:k + 1, k] + (1 - q) * V[0:k, k])
    else:
        def g(k):
            # snell envelope in the american case
            return np.maximum(K - S[0:k, k - 1], math.exp(-r * delta_t) * (q * V[1:k + 1, k] + (1 - q) * V[0:k, k]))

    # compute put prices at t_i
    for k in range(M, 0, -1):
        V[0:k, k - 1] = g(k)

    # return the price of the put at time t_0 = 0
    return V[0, 0]

# part b)
# computes the price of a put in the Black-Scholes model
def BlackScholes_EuPut(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(-d_1)
    C = K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(-d_2) -S_t * phi
    return C

# part c)
# test parameters
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = range(10, 501, 1)
K = 120

EuPutprices = np.zeros(501)
for i in M:
    EuPutprices[i] = CRR_AmEuPut(S_0, r, sigma, T, i, K, EU = 1)
BSprice = BlackScholes_EuPut(0, S_0, r, sigma, T, K)

plt.plot(np.arange(10,501),EuPutprices[10:], 'r', label = 'Binomial model price')
plt.plot(np.arange(10,501), BSprice * np.ones(491),'b', label = 'Black-Scholes price')
plt.xlabel('number of steps')
plt.ylabel('price')
plt.legend()
plt.show()

V_0 = CRR_AmEuPut(S_0, r, sigma, T, 500, K, EU = 0)
print('The price of the American put option for the test parameters is given by: ' + str(V_0))


