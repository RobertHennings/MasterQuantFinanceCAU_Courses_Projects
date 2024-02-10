### C-Exercise 36, SS 2023

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats


### Matrix Inversion Algorithm according to Section 5.2 in the lecture notes
def TriagMatrix_Inversion(alpha,beta,gamma,b):
    n = len(alpha)
    alpha_hat = np.zeros(n)
    b_hat = np.zeros(n)
    x = np.zeros(n)

    alpha_hat[0] = alpha[0]
    b_hat[0] = b[0]
    ### bringing the matrix on upper triangular form (forward recursion) according to p.75 in the lecture notes
    for i in range(1,n):
        alpha_hat[i] = alpha[i] - beta[i-1]*gamma[i]/alpha_hat[i-1]
        b_hat[i] = b[i] - b_hat[i-1]*gamma[i]/alpha_hat[i-1]

    ### solving the linear equation system(LES) according to (5.17) and (5.18)
    x[n-1] = b_hat[n-1]/alpha_hat[n-1]
    for i in range(n-2,-1,-1):
        x[i] = (b_hat[i]-beta[i] * x[i+1]) / alpha_hat[i]

    return x


def BS_EuCall_FiDi_CN(r, sigma, a, b, m, nu_max, T, K):
    ### setting the parameters needed for the recursion
    q = 2 * r / sigma ** 2
    delta_x = (b - a) / m
    delta_t = sigma ** 2 * T / (2 * nu_max)
    fidi_lambda = delta_t / delta_x ** 2

    ### range of underlying transformed stock prices
    x = np.arange(a, b + delta_x, delta_x)

    ### allocating memory
    w = np.zeros((m + 1, nu_max + 1))

    ### initial values equivalent to transformed payoff at maturity
    w[:, 0] = np.maximum(0, np.exp(x / 2 * (q + 1)) - np.exp(x / 2 * (q - 1)))

    ### setting the main and side diagonals of the tridiagonal matrix 'A_impl'
    alpha = np.ones(m-1) * (1+fidi_lambda)
    beta = np.ones(m-1)  * (-fidi_lambda/2)
    gamma = np.ones(m-1) * (-fidi_lambda/2)

    ### loop over columns of matrix/time
    for nu in range(1, nu_max + 1):

        ### loop over rows/underlying (transformed) stock price
        ### note that we do not change the top and bottom row which are equal to zero all the time (simplified boundary conditions)
        for j in range(1, m):
            ### calculating the right hand side of (5.21), can be seen as the explicit part of the CN-scheme
            w[j, nu] = fidi_lambda/2 * w[j - 1, nu - 1] + (1-fidi_lambda) * w[j, nu - 1] + fidi_lambda/2 * w[j + 1, nu - 1]

        ### boundary condition for right hand side (next time step)
        w[m-1,nu] = w[m-1,nu] + fidi_lambda/2 * (np.exp((q + 1) / 2 * b + (q + 1) ** 2 / 4 * nu * delta_t) - np.exp(
            (q - 1) / 2 * b + (q - 1) ** 2 / 4 * nu * delta_t))
        w[1:-1, nu] = TriagMatrix_Inversion(alpha, beta, gamma, w[1:-1, nu])

        ### boundary condition for explicit part of next iteration (could also be part of line 62) as in lecture notes
        w[m, nu] = np.exp((q + 1) / 2 * b + (q + 1) ** 2 / 4 * nu * delta_t) - np.exp(
            (q - 1) / 2 * b + (q - 1) ** 2 / 4 * nu * delta_t)

    ### retransfoming underlying stock prices
    S = K * np.exp(x[1:-1])

    ### transforming the solution of (5.1) into option prices
    V = K * w[1:-1, nu_max] * np.exp(-x[1:-1] / 2 * (q - 1) - sigma ** 2 / 2 * T * ((q - 1) ** 2 / 4 + q))
    return [S, V]


### test parameter
r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100

[S, V] = BS_EuCall_FiDi_CN(r, sigma, a, b, m, nu_max, T, K)


### BS-Formula
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    Call = S_t * scipy.stats.norm.cdf(d_1) - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return Call


V_BS = np.zeros(len(S))
### applying the BS-Formula to each underlying stock price
### note that we do set the stock prices only indirectly through the parameters a,b and m
for i in range(0, len(S)):
    V_BS[i] = EuCall_BlackScholes(0, S[i], r, sigma, T, K)

plt.plot(S, V, label='Price with finite difference scheme')
plt.plot(S, V_BS, label='Price with BS-Formula')
plt.legend()
plt.show()
