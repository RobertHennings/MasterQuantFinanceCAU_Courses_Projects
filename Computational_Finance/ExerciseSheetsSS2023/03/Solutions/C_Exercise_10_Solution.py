# C-Exercise 10, SS 2023

import math
import numpy as np
import scipy.stats


def Eu_Option_BS_MC(S0, r, sigma, T, K, M, f):
    # generate M samples
    X = np.random.normal(0, 1, M)
    ST = np.empty(len(X), dtype=float)
    Y = np.empty(len(X), dtype=float)

    # compute ST and Y for each sample
    for i in range(0, len(X)):
        ST[i] = S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * X[i])
        Y[i] = f(ST[i], K)

    # calculate V0
    VN_hat = np.mean(Y)
    # calculate V0
    V0 = math.exp(-r * T) * VN_hat

    # compute confidence interval
    epsilon = 1.96 * math.sqrt(np.var(Y) / M)
    c1 = math.exp(-r * T) * (VN_hat - epsilon)
    c2 = math.exp(-r * T) * (VN_hat + epsilon)
    return V0, VN_hat, c1, c2


# test parameters
S0 = 110
r = 0.04
sigma = 0.2
T = 1
M = 10000
K = 100


# european call
def g(x, K):
    return max(x - K, 0)


results = Eu_Option_BS_MC(S0, r, sigma, T, K, M, g)
V0 = results[0]
VN_hat = results[1]
c1 = results[2]
c2 = results[3]


# computes the price of a call in the Black-Scholes model
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C


V1 = BlackScholes_EuCall(0, S0, r, sigma, T, K)

print(
    'Price of European Call by use of plain Monte-Carlo simulation: ' + str(V0) + ', 95% confidence interval: [' + str(
        c1) + ',' + str(c2) + '].')
print('Exact price of European Call by use of the BS-formula: ' + str(V1))
