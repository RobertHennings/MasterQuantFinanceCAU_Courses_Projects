# C-Exercise 38, SS 2023

import numpy as np
import math
import matplotlib.pyplot as plt

def BS_AmPut_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K):
    return BS_AmPut_FiDi_General(r, sigma, a, b, m, nu_max, T, K, 0)


# This is the Code from p.77
def BS_AmPut_FiDi_General(r, sigma, a, b, m, nu_max, T, K, theta):
    # Compute delta_x, delta_t, x, lambda and set starting w
    q = (2 * r) / (sigma * sigma)
    delta_x = (b - a) / m
    delta_t = (sigma * sigma * T) / (2 * nu_max)
    lmbda = delta_t / (delta_x * delta_x)
    x = np.ones(m + 1) * a + np.arange(0, m + 1) * delta_x
    t = delta_t * np.arange(0, nu_max + 1)
    g_nu = np.maximum(np.exp(x * 0.5 * (q - 1)) - np.exp(x * 0.5 * (q + 1)), np.zeros(m + 1))
    w = g_nu[1:m]

    # Building matrix for t-loop
    lambda_theta = lmbda * theta
    diagonal = np.ones(m - 1) * (1 + 2 * lambda_theta)
    secondary_diagonal = np.ones(m - 2) * (- lambda_theta)
    b = np.zeros(m - 1)

    # t-loop as on p.77.
    for nu in range(0, nu_max):
        g_nuPlusOne = math.exp((q + 1) * (q + 1) * t[nu + 1] / 4.0) * np.maximum(np.exp(x * 0.5 * (q - 1))
                                                                                 - np.exp(x * 0.5 * (q + 1)),
                                                                                 np.zeros(m + 1))
        b[0] = w[0] + lmbda * (1 - theta) * (w[1] - 2 * w[0] + g_nu[0]) + lambda_theta * g_nuPlusOne[0]
        b[1:m - 2] = w[1:m - 2] + lmbda * (1 - theta) * (w[2:m - 1] - 2 * w[1:m - 2] + w[0:m - 3])
        b[m - 2] = w[m - 2] + lmbda * (1 - theta) * (g_nu[m] - 2 * w[m - 2] + w[m - 3]) + lambda_theta * g_nuPlusOne[m]

        # Use Brennan-Schwartz algorithm to solve the linear equation system
        w = solve_system_put(diagonal, secondary_diagonal, secondary_diagonal, b, g_nuPlusOne[1:m])

        g_nu = g_nuPlusOne

    S = K * np.exp(x[1:m])
    v = K * w * np.exp(- 0.5 * x[1:m] * (q - 1) - 0.5 * sigma * sigma * T * ((q - 1) * (q - 1) / 4 + q))

    return S, v


# This is the code from Lemma 5.3
def solve_system_put(alpha, beta, gamma, b, g):
    n = len(alpha)
    alpha_hat = np.zeros(n)
    b_hat = np.zeros(n)
    x = np.zeros(n)

    alpha_hat[n - 1] = alpha[n - 1]
    b_hat[n - 1] = b[n - 1]
    for i in range(n - 2, -1, -1):
        alpha_hat[i] = alpha[i] - beta[i] / alpha_hat[i + 1] * gamma[i]
        b_hat[i] = b[i] - beta[i] / alpha_hat[i + 1] * b_hat[i + 1]
    x[0] = np.maximum(b_hat[0] / alpha_hat[0], g[0])
    for i in range(1, n):
        x[i] = np.maximum((b_hat[i] - gamma[i - 1] * x[i - 1]) / alpha_hat[i], g[i])
    return x


# C-Exercise 06, SS 2023
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
        if k == 1:
            V[0, 0] = g(k)
        else:
            V[0:k, k - 1] = g(k)

    # return the price of the put at time t_0 = 0
    return V[0, 0]



# Initialize the test parameters given in the exercise.
r = 0.05
sigma = 0.2
a = - 0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 95

initial_stock, option_prices = BS_AmPut_FiDi_Explicit(r, sigma, a, b, m, nu_max,
                                                      T, K)
exercise9 = np.zeros(len(initial_stock))
for j in range(0, len(initial_stock)):
    exercise9[j] = CRR_AmEuPut(initial_stock[j], r, sigma, T, 500, K, 0)

# Compute the absolute difference between the approximation and the option prices from exercise 9
absolute_errors = np.abs(option_prices - exercise9)

# Compare the results by plotting the absolute error.
plt.plot(initial_stock, absolute_errors)
plt.xlabel('initial stock price')
plt.ylabel('absolute difference')
plt.title('The absolute difference between the finite difference approximation with the explicit scheme and the'
          ' option prices from exercise 9')
plt.show()
