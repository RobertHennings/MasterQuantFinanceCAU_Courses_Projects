# Sheet: 03, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import numpy as np
from scipy.stats import norm
import scipy.stats as st
# C-Exercise 10
# Compute European options price via Monte Carlo Simulation, EU Call
# We need to simulate a standard normally distributed random variable
def f(x):
    return np.maximum(x-100, 0)
S0 = 110
r = 0.04
sigma = 0.2
T = 1
M = 10000
K = 100
t = 0

def Eu_Option_BS_MC(S0: float, r: float, sigma: float, T: int, K: float, M: int, f: callable) -> float:
    """Compute the European Call option price using Monte Carlo simulated stock prices and the
       respective confidence interval of the simulated Option values

    Args:
        S0 (float): Initial stock price
        r (float): Interest rate
        sigma (float): Volatility
        T (int): Time horizon
        K (float): Strike price
        M (int): Discretization of the Time horizon
        f (callable): Payoff function, either Call or Put type

    Returns:
        float: EU Call price obtained by Monte Carlo Simulation, Respective 95% Confidence interval
    """
    # We have the given formula from the BSM world, in which we want to simulate X with law N(0,1) under Q
    # Finally we arrive at M simulations of S(T) onto which we apply f(x) and discount to get V(0)

    # Simulate X ~ N(0,1) via Monte Carlo
    # np.random.seed(22)
    X = np.random.normal(loc=0, scale=1, size=M)

    # Compute terminal stock price based on simulated Xs
    S_T = S0 * np.exp((r - sigma**2/2) * T + sigma * np.sqrt(T) * X)
    # Apply the given Call function to the terminal stock prices
    payoffs = f(x=S_T)
    # Finally discount the payoffs of the call
    value = np.exp(-r * T) * payoffs
    # also get the confidence interval here
    quantiles = st.t.interval(alpha=0.95, df=len(value)-1, loc=np.mean(value), scale=st.sem(value))
    # As we take from the lecture notes f.2.1) take here the emprical mean for the expected value
    value = np.mean(value)
    return value, quantiles


# Compare with the BSM price from Exercise Sheet 02, EU Call
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    # Compute static values first
    # Compute d1 and d2
    d1 = (np.log(S_t/K) + (((sigma**2)/2)+r) * (T-t)) / (sigma * (np.sqrt(T-t)))
    d2 = d1 - sigma * (np.sqrt(T-t))

    EuCall = S_t * norm.cdf(d1) - K * (np.exp(-r * (T-t)) * norm.cdf(d2))

    return EuCall



# Compare the MC simulated value and the BSM value, return also the confidence intervals
MC_sim, MC_quantiles = Eu_Option_BS_MC(S0=S0, r=r, sigma=sigma, T=T, K=K, M=M, f=f)
BSM_CallEU_P = BlackScholes_EuCall(t=t, S_t=S0, r=r, sigma=sigma, T=T, K=K)

print("The EU Call Option price obtained from the MC Simulation is: ", MC_sim, "with respective 95% confidence interval: ", MC_quantiles)
print("The BSM EU Call Option price price is: ", BSM_CallEU_P)
