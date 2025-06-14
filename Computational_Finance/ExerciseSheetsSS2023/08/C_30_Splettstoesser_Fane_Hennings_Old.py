# Sheet: 08, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
# C-Exercise 30
# Valuation of European options in the Heston model using the Euler method

def Heston_EuCall_MC_Euler(S0: float, r: float, gamma_0: float,
                           kappa: float, lambda_: float, sigma: float,
                           T: int, g: callable, M: int, m: int) -> list:
    """Fair value of European options in the Heston model via the
       Monte-Carlo method using M samples together with the
       95%- confidence interval. Paths are approximated by the
       Euler method with a grid of m equidistant points in time.

    Args:
        S0 (float): Initial underlying (stock) price
        r (float): Interest rate
        gamma_0 (float): _description_
        kappa (float): _description_
        lambda_ (float): _description_
        sigma (float): Volatility
        T (int): Time Horizon
        g (callable): payoff function (call or put)
        M (int): Number of simulations
        m (int): _description_

    Returns:
        list: Fair value of European option, upper and lower
              95%-confidence bound
    """
    # Check for datatype
    for var in [r, sigma, S0, gamma_0, kappa, lambda_]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, M, m]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    # approximate the paths by the Euler method, see the defined algorithm on page 54, 55 in the script
    # Recall the Heston model assumes time variant volatility
    # Initalize the time delta steps
    delta_t = T / m # watch out small M here
    # Create matrices that will hold the values
    stock_T_mat = np.zeros(M)
    value_mat = np.zeros(M)

    # Generate random numbers, now here both for the stock price paths and the volatility paths
    z_stock = np.random.normal(loc=0, scale=1, size=M)
    z_gamma = np.random.normal(loc=0, scale=1, size=M)

    # loop through to compute the values
    for i in range(0, M, 1):
        # save the stock paths and the gamma paths alongside
        stock_mat = np.zeros(m)
        gamma_mat = np.zeros(m)

        # Set the intial starting values for the paths
        stock_mat[0] = S0
        gamma_mat[0] = gamma_0

        # Now get to the Euler discretization wher we need to tweak the Heston Model
        # that also incorporates timevariant volatilty
        for j in range(1, m, 1):
            gamma_mat[j] = gamma_mat[j-1] + kappa + (lambda_ - gamma_mat[j-1]) * delta_t + sigma * np.sqrt(gamma_mat[j-1] * delta_t) * z_gamma[j]
            stock_mat[j] = stock_mat[j-1] * np.exp((r - 0.5  * gamma_mat[j-1]) * delta_t + np.sqrt(gamma_mat[j-1] * delta_t) * z_stock[j])

        # Compute Option payoff
        stock_T_mat[i] = stock_mat[-1]
        value_mat[i] = np.maximum(g(stock_mat[-1]), 0)

    # Compute option price and confidence interval
    V0 = np.mean(value_mat) * np.exp(-r * T)
    std_err = np.std(value_mat) / np.sqrt(M)
    c1 = V0 - 1.96 * std_err
    c2 = V0 + 1.96 * std_err

    return [V0, c1, c2]


# Test the function for given parameters
S0 = 100.0
r = 0.05
gamma_0 = 0.2**2
kappa = 0.5
lambda_ = 2.5
sigma = 1.0
T = 1
K = 100.0
g = lambda x: np.maximum(x-K, 0) # Call Option
M = 10000
m = 250

V_0, C_l, C_U = Heston_EuCall_MC_Euler(S0=S0, r=r, gamma_0=gamma_0, kappa=kappa,
                                       lambda_=lambda_, sigma=sigma, T=T, g=g, M=M, m=m)
print(f"Fair value of a european option (Call) with strike: {K} using the Euler Method: {V_0} with lower 95%-confidence bound: {C_l} and upper 95%-confidence bound: {C_U}")
