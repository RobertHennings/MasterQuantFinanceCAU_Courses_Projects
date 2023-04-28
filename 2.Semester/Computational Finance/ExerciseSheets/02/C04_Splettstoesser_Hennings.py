# Sheet: 02, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
# C-Exercise 04
# a)
# Price Call Options with the CRR model
# Hint: Use the results from C-Exercise 01 and the process from T-Exercise 05.
# EU Call
def CRR_EuCall(S_0: float, r: float, sigma: float, T: int, M: int, K: float) -> float:
    """Compute the value of a European Call Option approximated by the Cox-Ross-Rubinstein Model

    Args:
        S_0 (float): Initial underlying (stock) price
        r (float): Interest rate
        sigma (float): Volatility
        T (int): Time Horizon
        M (int): Discretization of the Time Horizon into M equal spaced steps
        K (float): Strike price of option

    Returns:
        float: Initial Option Value V_0
    """
    for var in [S_0, r, sigma, M, K]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, M]:
        if not isinstance(var, int):
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
            payoff[j, i] = np.maximum(stock[j, i] - K, 0)
    # determine the payoffs at maturity, fill the last column with the payoff values
    # and compute backwards in time the option value
    value[:, M] = payoff[:, M]

    # Calculate reversed the value
    for i in range(M-1, -1, -1):
        for j in range(0, i+1):
          value[j, i] = np.exp(-r*delta_t) * (q * value[j+1, i+1] + (1-q) * value[j, i+1])
    # return stock, payoff, value
    return value[0, 0]


# Test function
CRR_EuCall(S_0=1.0, r=0.05, sigma=0.3, T=3, M=3, K=1.2)
# 0.20635464832858255

# b)
# BS-Formula for European call options
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    # Compute static values first
    # Compute d1 and d2
    d1 = (np.log(S_t/K) + (((sigma**2)/2)+r) * (T-t)) / (sigma * (np.sqrt(T-t)))
    d2 = d1 - sigma * (np.sqrt(T-t))

    EuCall = S_t * norm.cdf(d1) - K * (np.exp(-r * (T-t)) * norm.cdf(d2))

    return EuCall

# Test function
BlackScholes_EuCall(t=0, S_t=1.0, r=0.05, sigma=0.3, T=3, K=1.2)
# 0.19232384812201503


# c) Compare the CRR Model with the BSM Formula over several strikes
# Set testing parameters
S_0 = 100.00
t = 0
r = 0.03
sigma = 0.3
T = 1
M = 100
# Keep strikes variable and compute option values
strikes = np.arange(70.00, 201.00, 1.00)

model_error = np.empty(len(strikes), dtype=float)
bs_price = np.empty(len(strikes), dtype=float)
crr_price = np.empty(len(strikes), dtype=float)


for i in range(0,len(strikes),1):
    crr_price[i] = CRR_EuCall(S_0=S_0, r=r, sigma=sigma, T=T, M=M, K=strikes[i])
    bs_price[i] = BlackScholes_EuCall(t=t, S_t=S_0, r=r, sigma=sigma, T=T, K= strikes[i])
    model_error[i] = bs_price[i] - crr_price[i]


# Compare the Model Error against the BSM price
plt.plot(strikes, model_error, color="#603086")
plt.ylabel("Model Err (BSM - CRR)")
plt.xlabel("Strike Prices", fontsize=6)
plt.axhline(y=0, color="black")
plt.title("Model Error vs. Strike", fontsize=14)
plt.xticks(rotation=45, fontsize=6)
plt.show()

# Compare the CRR and BSM price against the strike
plt.plot(strikes, bs_price, color="#603086", label="BSM")
plt.plot(strikes, crr_price, color="black", label="CRR")
plt.ylabel("BSM vs. CRR Model Price")
plt.xlabel("Strikes", fontsize=6)
plt.axvline(x=S_0, color="red")
plt.text(S_0+2, 21.0, "Spot price S_0")
plt.title("BSM Model Price vs. CRR Model Price EU Call", fontsize=14)
plt.xticks(rotation=45, fontsize=6)
plt.legend()
plt.show()
