# Sheet: 04, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st

# C-Exercise 13
# Using control variables to reduce the variance of MC-estimators

def BS_EuOption_MC_CV(S0: float, r: float, sigma: float, T: int, K: int, M: int) -> float:
    """Compute the European Call option price in a Quanto style
       using Monte Carlo simulated stock prices

    Args:
        S0 (float): Initial stock price
        r (float): Interest rate
        sigma (float): Volatility
        T (int): Time horizon
        K (float): Strike price
        M (int): Discretization of the Time horizon
        f (callable): Payoff function, either Call or Put type

    Returns:
        float: EU Call price for Quanto style obtained by Monte Carlo Simulation
               via the Control variate technique
    """
    # Idea: We want to simulate in parallel the normal EU Call and the Quanto Call
    # Finally we want to compute the estimator V_CV according to formula on page 19 in the script

    # Implemet the normal EU Call payoff function
    def f(x, K):
        return np.maximum(x-K, 0)
    # Therefore simulate two random variables X as before and Y additionally
    # one for each Option type, both follow the same law: (X_n, Y_n)
    # estimator should equal: E(f(X)) = E(Y) + E(f(X) − Y)
    # In which f(X) is the Quanto Call part and Y is our Normal EU Call
    # np.random.seed(22)
    X = np.random.normal(loc=0, scale=1, size=M)
    Y = np.random.normal(loc=0, scale=1, size=M)

    # Save now for both option types the terminal stock prices as well as the payoffs
    S_T_norm = np.zeros(M)
    S_T_quanto = np.zeros(M)
    # save the payoffs
    payoff_norm = np.zeros(M)
    payoff_quanto = np.zeros(M)

    # Next just as usual compute the terminal stock price S_T and the payoff for each option type
    for ind in range(M):
        # First the normal EU Call Option: Y
        S_T_norm[ind] = S0 * np.exp((r - sigma**2/2) * T + sigma * np.sqrt(T) * Y[ind])
        payoff_norm[ind] = f(x=S_T_norm[ind], K=K)
        # Next the Quanto Call Option: f(X)
        S_T_quanto[ind] = S0 * np.exp((r - sigma**2/2) * T + sigma * np.sqrt(T) * X[ind])
        payoff_quanto[ind] = f(x=S_T_quanto[ind], K=K) * S_T_quanto[ind]  # here edit the payoff to the Quanto version

    # Next we are asked to work with the beta factor that can reduce the variance of the simulation
    # Beta factor was given in the exercise sheet
    # Cov of Normal EU Call and Quanto Call Payoffs divided by the Var of the normal EU Call
    beta = np.cov(payoff_quanto, payoff_norm)[0][1] / np.var(payoff_norm)
    # Finally we want to discount and arrive at the initial Option value
    # Watch out: apply slightly different formula, see page 19: E(f(X)) = E(Y) + E(f(X) − Y)
    # Here with beta it would be: E(f(X)) = E(Y * beta) + E(f(X) − Y * beta)
    V_CV = np.exp(-r * T) * (np.mean(beta * payoff_norm) + np.mean(payoff_quanto - beta * payoff_norm))
    return V_CV

# Finally compare the results: Introduced CV MC against the normal MC fom C 10
# Be careful we alos have to edit the payof function here to fit the case of the Quanto Option style
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
    S_T = np.zeros(M)
    payoffs = np.zeros(M)
    # Compute terminal stock price based on simulated Xs
    for ind in range(M):
        S_T[ind] = S0 * np.exp((r - sigma**2/2) * T + sigma * np.sqrt(T) * X[ind])
        payoffs[ind] = f(x=S_T[ind], K=K) * S_T[ind]  # edit to match the Quanto option style here

    value = np.exp(-r * T) * payoffs
    # also get the confidence interval here
    quantiles = st.t.interval(alpha=0.95, df=len(value)-1, loc=np.mean(value), scale=st.sem(value))
    # As we take from the lecture notes f.2.1) take here the emprical mean for the expected value
    value = np.mean(value)
    return value, quantiles



S0 = 100.0
r = 0.05
sigma = 0.3
T = 1
K = 110.0
M = 100000
# For the normal simulation
def f(x, K):
    return np.maximum(x-K, 0)


# Compare
print(f"The Covariate Variable method yields a value of: {BS_EuOption_MC_CV(S0, r, sigma, T, K, M)}\
      whereas the normal MC simulation yields: {Eu_Option_BS_MC(S0, r, sigma, T, K, M, f)[0]}")
