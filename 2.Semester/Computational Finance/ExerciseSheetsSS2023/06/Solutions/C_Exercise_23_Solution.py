# C-Exercise 17, SS 2022

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# compute Black-Scholes price by integration
def BS_Price_Int(S0, r, sigma, T, f):
    # define integrand as given in the exercise
    def integrand(x):
        return 1 / math.sqrt(2 * math.pi) * f(
            S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * x)) * math.exp(-r * T) * math.exp(
            -1 / 2 * math.pow(x, 2))

    # perform integration
    I = integrate.quad(integrand, -np.inf, np.inf)
    # return value of the integration
    return I[0]

# compute greeks of an european option in the Black-Scholes model
def BS_Greeks_num(r, sigma, S0, T, g, eps):
    # Evaluation price function for initial parameters and by epsilon augmented parameters
    V0 = BS_Price_Int(S0, r, sigma, T, g)
    V_Delta = BS_Price_Int((1 + eps) * S0, r, sigma, T, g)
    V_vega = BS_Price_Int(S0, r, (1 + eps) * sigma, T, g)
    V_gamma1 = V_Delta
    V_gamma2 = BS_Price_Int((1 - eps) * S0, r, sigma, T, g)

    # computation of the greeks
    Delta = (V_Delta - V0) / (eps * S0)
    vega = (V_vega - V0) / (eps * sigma)
    gamma = (V_gamma1 - 2 * V0 + V_gamma2) / (math.pow(eps * S0, 2))
    return Delta, vega, gamma


# test parameters
r = 0.05
sigma = 0.3
T = 1
S0 = range(60, 141, 1)
eps = 0.001


# define payoff function for call with strike 100
def g(x):
    return max(x - 110, 0)


# allocate vector for greeks
Delta = np.empty(81, dtype=float)
vega = np.empty(81, dtype=float)
gamma = np.empty(81, dtype=float)

for i in range(0, len(S0)):
    result = BS_Greeks_num(r, sigma, S0[i], T, g, eps)
    Delta[i] = result[0]
    vega[i] = result[1]
    gamma[i] = result[2]

# plot each vector in a seperate subplot
plt.clf()
plt.subplot(2, 2, 1)
plt.plot(S0, Delta)
plt.xlabel('S0')
plt.ylabel('Delta')
plt.title('Delta')

plt.subplot(2, 2, 2)
plt.plot(S0, vega)
plt.xlabel('S0')
plt.ylabel('vega')
plt.title('vega')

plt.subplot(2, 2, 3)
plt.plot(S0, gamma)
plt.xlabel('S0')
plt.ylabel('gamma')
plt.title('gamma')

plt.show()
