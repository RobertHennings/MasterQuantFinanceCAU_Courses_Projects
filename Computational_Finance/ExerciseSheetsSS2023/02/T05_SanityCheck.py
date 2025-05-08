import numpy as np
from scipy.stats import norm
import datetime as dt
import math
import matplotlib.pyplot as plt
# C-Exercise 04
# a)
# Price Call Options with the CRR model
# Hint: Use the results from C-Exercise 01 and the process from T-Exercise 05.
# EU Call
def CRR_AmEU(S_0: float, r: float, sigma: float, T: int, M: int, K: float, EU: int, type_: str) -> float:
    """Compute the value of a European Call Option approximated by the Cox-Ross-Rubinstein Model

    Args:
        S_0 (float): Initial underlying (stock) price
        r (float): Interest rate
        sigma (float): Volatility
        T (int): Time Horizon
        M (int): Discretization of the Time Horizon into M equal spaced steps
        K (float): Strike price of option
        EU (int): Either 0 for American Put Option or 1 for European Put Option type
        type_ (str): Either "call" for call or "put" for put option 

    Returns:
        float: Initial Option Value V_0
    """
    for var in [S_0, r, sigma, M, K]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, M, EU]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    for var in [type_]:
        if not isinstance(var, str):
            raise TypeError(f"{var} not type int")
    # Compute static values first
    # Delta t
    delta_t = (T/M)
    # Beta
    beta = (1/2) * (np.exp(-r*delta_t) + np.exp((r+sigma**2)*delta_t))
    # Up factor of underlying
    u = beta + np.sqrt(beta**2-1)
    # Down factor of underlying
    d = (1/u)
    # up probability q and down prob. (1-q)
    q = (np.exp(r*delta_t) - d) / (u-d)

    # Create matrices that will be filled
    stock = np.zeros((M+1, M+1))
    payoff = np.zeros((M+1, M+1))
    value = np.zeros((M+1, M+1))

    # First compute underlying evolution and option payoff in each period
    for i in range(0, M+1):
        for j in range(0, i+1):
            stock[j, i] = S_0 * u**j * d**(i-j)
            # payoff of either put or call type
            if type_ == "put":
                payoff[j, i] = np.maximum(K - stock[j, i], 0)
            else:
                payoff[j, i] = np.maximum(stock[j, i] - K, 0)

    # determine the payoffs at maturity
    value[:, M] = payoff[:, M]

    # Calculate reversed the value
    for i in range(M-1, -1, -1):
        for j in range(0, i+1):
            if EU == 0:  # AM
                value[j, i] = np.maximum(payoff[j, i], np.exp(-r*Delta_t) * (q * value[j+1, i+1] + (1-q) * value[j, i+1]))
            else:  # EU
                value[j, i] = np.exp(-r*Delta_t) * (q * value[j+1, i+1] + (1-q) * value[j, i+1])

    # return stock, payoff, value
    return value[0, 0]


# Test function
CRR_AmEU(S_0=1.0, r=0.05, sigma=np.sqrt(0.3), T=3, M=3, K=1.2, EU=0, type_="put")
# 0.4597937866305617

# b) Manual calculations
# t_i = 1
(0.2157 - 0.6580) / 1.2720

(0.46 + 0.3477) / np.exp(0.05)
# t_i = 2
# Upper path
(0-3.747) / (1.8211 *1.2720)

(0.2157 + (1.6176*1.8211)) / np.exp((0.05*2))
# Lower path
(0.3747 - 0.8985) / (0.5491 * 1.2720)

(0.6580 + (0.7499*0.5491)) / np.exp((0.05*2))