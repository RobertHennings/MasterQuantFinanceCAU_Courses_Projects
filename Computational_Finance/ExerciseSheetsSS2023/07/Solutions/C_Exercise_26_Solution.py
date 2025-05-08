# C-Exercise 26, SS 2023

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# computes the price of the American Option from T-Ex.20
def CRR_AmOption(S_0, T, M):
    r = 0
    sigma = math.sqrt(2)
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    q = (math.exp(r * delta_t) - d) / (u - d)

    # allocate matrix S
    S = np.zeros((M + 1, M + 1))

    # fill matrix S with stock prices
    # going down in the matrix means that the stock price goes up
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)

    # payoff for a vector of stock prices according to formula (1)
    def g(S_vector):
        payoff = np.zeros(len(S_vector))
        for i in range(0,len(S_vector)):
            if S_vector[i] < 1:
                payoff[i] = 4 * math.pow(S_vector[i],3/4)
            else:
                payoff[i] = 3 * math.sqrt(S_vector[i]) + math.pow(S_vector[i],3/2)
        return payoff

    # allocate memory for option prices
    V = np.zeros((M + 1, M + 1))

    # option price at maturity, formula (1.16) in lecture notes
    V[:, M] = g(S[:, M])

    # Loop goes backwards through the columns
    for i in range(M - 1, -1, -1):
        # backwards recursion for European option, formula (1.14)
        V[0:(i + 1), i] = np.exp(-r * delta_t) * (q * V[1:(i + 2), i + 1] + (1 - q) * V[0:(i + 1), i + 1])

        # compare 'exercising the option', i.e. immediate payoff,
        # with 'waiting', i.e. the expected (discounted) value of the next timestep which corresponds to the price
        # of a european option, formula (1.15)
        V[0:(i + 1), i] = np.maximum(g(S[0:(i + 1), i]), V[0:(i + 1), i])

    # first entry of the matrix is the initial option price
    return V[0, 0]

# computes initial option price according to formula (2) in T-Ex.20
def CRR_AmOptionDirect (S_0, T):
    if S_0 < math.exp(-T):
        if S_0 < 1:
            V_0 = 4 * math.pow(S_0, 3 / 4)
        else:
            V_0 = 3 * math.sqrt(S_0) + math.pow(S_0, 3 / 2)
    else:
        V_0 =  math.exp(-T/4) * 3 * math.sqrt(S_0) + math.exp(T * 3/4) * math.pow(S_0, 3/2)
    return V_0

# test parameters
S_0 = 1
T = 1
M = 500
t = 0


print('Price in the CRR-Model: ' + str(CRR_AmOption(S_0, T, M)))
print('Price with explicit formula: ' + str(CRR_AmOptionDirect (S_0, T)))
