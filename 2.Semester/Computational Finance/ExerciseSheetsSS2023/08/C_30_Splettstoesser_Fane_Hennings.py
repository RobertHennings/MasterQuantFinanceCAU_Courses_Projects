# Sheet: 08, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
# C-Exercise 30
# Valuation of European options in the Heston model using the Euler method to simulate full underlying stock prices

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

    # Initialize arrays to store option prices and paths
    value_mat = np.zeros(M)
    ST_mat = np.zeros(M)

    def SimPath_Ito_Euler(X0, a, b, T, m, N):
        # Initalize the time delta steps
        delta_t = (T / m) # watch out small M here
        # Create the delta_ws
        delta_w = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=(M, m)) # watch out: Use different Vola here
        # Our approximation should be Y, we need to store its values accordingly
        Y = np.zeros((M, m))
        Y[:, 0] = X0
        # Generate the loop and compute the values according to the algo on page 55
        for i in range(1, m):
            Y[:, i] = Y[:, i - 1] + a(Y[:, i - 1], (i - 1) * delta_t) * delta_t + b(Y[:, i - 1], (i - 1) * delta_t) * delta_w[:, i - 1]

        return Y
    # Create "sufficiently smooth” coefficients a(x,t),b(x,t) that we will plug into the function
    # depends on what model is observed, here we set a and b to fullfill the Heston Model coefficients
    def a(x, t):
        return kappa - lambda_ * x

    def b(x, t):
        return np.sqrt(x) * sigma

    for i in range(M):
        # Simulate gamma paths using the modified SimPath_Ito_Euler function
        gamma = SimPath_Ito_Euler(gamma_0, a, b, T, m, 1)[0]

        # Generate random numbers for price process
        Z_S = np.random.normal(0, 1, m)

        S = np.zeros(m)

        # Initialize initial value for price
        S[0] = S0

        # Simulate price path using Euler discretization
        for j in range(1, m):
            S[j] = S[j - 1] * np.exp((r - 0.5 * gamma[j - 1]) * delta_t + np.sqrt(gamma[j - 1] * delta_t) * Z_S[j])

        # Compute option payoff at maturity
        ST = S[-1]
        ST_mat[i] = ST
        value_mat[i] = np.maximum(g(ST), 0)

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
g = lambda x: np.maximum(x-K, 0)
M = 10000
m = 250

V_0, C_l, C_U = Heston_EuCall_MC_Euler(S0=S0, r=r, gamma_0=gamma_0, kappa=kappa,
                                       lambda_=lambda_, sigma=sigma, T=T, g=g, M=M, m=m)
print(f"Fair value of a european option (Call) with strike: {K} using the Euler Method: {V_0} with lower 95%-confidence bound: {C_l} and upper 95%-confidence bound: {C_U}")