# C-Exercise 30, SS 2023
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import cmath

def Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m):
    Delta_t = T/m
    Delta_W1 = np.random.normal(0,math.sqrt(Delta_t), (M, m))
    Delta_W2 = np.random.normal(0, math.sqrt(Delta_t), (M, m))

    #Initialize matrix which contains the process values
    S = np.zeros((M, m+1))
    gamma = np.zeros((M, m+1))

    #Assign first column starting values
    S[:, 0] = S0 * np.ones(M)
    gamma[:, 0] = gamma0 * np.ones(M)

    #Recursive column-wise simulation according to the algorithms in Section ?.? using one vector of Brownian motion increments
    for i in range(0, m):
        gamma[:, i+1] = np.maximum(gamma[:, i] + (kappa - lmbda * gamma[:, i]) * Delta_t + sigma_tilde * np.sqrt(gamma[:, i]) * Delta_W1[:, i],0)
        S[:,i+1] = S[:,i] + r * S[:, i] * Delta_t + S[:, i] * np.sqrt(np.maximum(gamma[:, i], 0)) * Delta_W2[:, i]

    payoff = g(S[:,-1])
    MC_estimator = math.exp(-r * T) * payoff.mean()
    epsilon = math.exp(-r * T) * (1.96 * math.sqrt(np.var(payoff, ddof=1) / M))
    c1 = MC_estimator - epsilon
    c2 = MC_estimator + epsilon
    return MC_estimator, c1, c2

#to measure how good the MC simulation is, we compute the true value with integral transforms, see lecture notes chapter 7
def Heston_EuCall_Laplace(S0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R):
    # Laplace transform of the function f(x) = (e^(xp) - K)^+ (cf. (7.6))
    def f_tilde(z):
        if np.real(z) > 0:
            return np.power(K, 1 - z) / (z * (z - 1))
        else:
            print('Error')

    # Characteristic function of log(S(T)) in the Heston model (cf. (7.8))
    def chi(u):
        d = cmath.sqrt(
            math.pow(lmbda, 2) + math.pow(sigma_tilde, 2) * (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))))
        n = cmath.cosh(d * T / 2) + lmbda * cmath.sinh(d * T / 2) / d
        z1 = math.exp(lmbda * T / 2)
        z2 = (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * cmath.sinh(d * T / 2) / d
        v = cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T)) * cmath.exp(
            2 * kappa / math.pow(sigma_tilde, 2) * cmath.log(z1 / n)) * cmath.exp(-nu0 * z2 / n)
        return v

    # integrand for the Laplace transform method (cf. (7.9))
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # integration to obtain the option price (cf. (7.9))
    V0 = integrate.quad(integrand, 0, 50)
    return V0[0]

if __name__ == '__main__':
    #Testing Parameters
    S0 = 100
    r = 0.05
    gamma0 = 0.2**2
    kappa = 0.5
    lmbda = 2.5
    sigma_tilde = 1
    T = 1
    M = 10000
    m = 250

    def g(x):
        return np.maximum(x - 100, 0)

    V0, c1, c2 = Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m)
    Heston_value = Heston_EuCall_Laplace(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, 100, 1.2)
    print("The option price is: " + str(Heston_value))
    print("The MC estimate is: " + str(V0))
    print("95% confidence interval: [" + str(c1) + ',' + str(c2) + "].")
