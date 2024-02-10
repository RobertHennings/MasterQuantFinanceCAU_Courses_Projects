# C-Exercise 01, SS 2023

import math
import numpy as np


# computes the stock price in the CRR model
def CRR_stockEU_AM_Put(S_0, r, sigma, T, M, K, EU):
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    q = (np.exp(r * delta_t) - d) / (u-d)
    print(q)
    # allocate matrix S
    S = np.zeros((M + 1, M + 1))
    value = np.zeros((M + 1, M + 1))
    exp_value = np.zeros((M + 1, M + 1))
    # fill matrix S with stock prices
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)
            value[j - 1, i - 1] = max(K - S[j - 1, i - 1], 0)
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            if i-1 <3:
                if EU == 1:
                    print(f"V_{j-1}_{i-1} = q* V_{j}_{i} + (1-q)* V_{j-1}_{i}")
                    print("q* ", value[j, i], " +(1-q)* ", value[j-1, i])
                    exp_value[j - 1, i - 1] = np.exp(-r * delta_t) * (q * value[j, i] + (1-q) * value[j-1, i])
                else:
                    exp_value[j - 1, i - 1] = max(value[j - 1, i - 1], np.exp(-r * delta_t) * (q * value[j, i] + (1-q) * value[j-1, i]))
            
    return S, value, exp_value

def CRR_stockEU_AM_Call(S_0, r, sigma, T, M, K, EU):
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    q = (np.exp(r * delta_t) - d) / (u-d)
    print(q)
    # allocate matrix S
    S = np.zeros((M + 1, M + 1))
    value = np.zeros((M + 1, M + 1))
    exp_value = np.zeros((M + 1, M + 1))
    # fill matrix S with stock prices
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)
            value[j - 1, i - 1] = max(S[j - 1, i - 1] - K, 0)
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            if i-1 <3:
                if EU == 1:
                    print(f"V_{j-1}_{i-1} = q* V_{j}_{i} + (1-q)* V_{j-1}_{i}")
                    print("q* ", value[j, i], " +(1-q)* ", value[j-1, i])
                    exp_value[j - 1, i - 1] = np.exp(-r * delta_t) * (q * value[j, i] + (1-q) * value[j-1, i])
                else:
                    exp_value[j - 1, i - 1] = max(value[j - 1, i - 1], np.exp(-r * delta_t) * (q * value[j, i] + (1-q) * value[j-1, i]))
            
    return S, value, exp_value

# test parameters
S_0 = 1.0
r = 0.05
# sigma = np.sqrt(0.3)
sigma = 0.3
T = 3
M = 3
K = 1.2
EU = 1


CRR_stockEU_AM_Put(S_0, r, sigma, T, M, K, EU)[2]
# 0.3747456

CRR_stockEU_AM_Call(S_0, r, sigma, T, M, K, EU)[2]
# 0.08258739