# Sheet: 08, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# C-Exercise 29
# Calculating the Delta/Hedge of a European option via MC methods, infinitesimal perturbation
# Calcuate the Hedging Position phi1_t

def EuOptionHedge_BS_MC_IP(St: float, r: float, sigma: float,
                           g: callable, T: int, t: int, N: int,
                           seed: int, set_seed: bool) -> float:
    """_summary_

    Args:
        St (float): Current underlying (stock) price
        r (float): Interest rate
        sigma (float): Volatility
        g (callable): payoff function (call or put)
        T (int): Time Horizon
        t (int): Current time step
        N (int): _description_
        seed (int): Seed for reproducability
        set_seed (bool): Indicate if reproducability desired for comparison

    Raises:
        TypeError: Check for datatype float
        TypeError: Check for datatype int
        TypeError: Check for datatype bool

    Returns:
        float: Delta/Hedge position of a European option via MC methods,
               infinitesimal perturbation
    """
    # start of by creating the values and empty matrices
    # need N standard normally distributed random variables, see page 52 for the specification of X
    # Check for Data type
    for var in [r, sigma, St]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, t, N, seed]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    for var in [set_seed]:
        if not isinstance(var, bool):
            raise TypeError(f"{var} not type bool")
    if set_seed:
        np.random.seed(seed)
    X = np.random.normal(loc=0, scale=1, size=N)
    # empty matrix to store the stock prices at the terminal T
    S_T = np.zeros(N)
    Zeta = np.zeros(N)

    # next simulate and compute the stock prices and the Z_primes directly by using formula
    for i in range(0, N, 1):
        # Save the terminal stock prices in dependence of X
        S_T[i] = St * np.exp((r - ((sigma**2)/2)) * (T-t) + sigma * np.sqrt((T-t)) * X[i])
        # Store the Z_primes, we apply the formula from page 52 directly
        Zeta[i] = np.exp((-(sigma**2) / 2) * (T-t) + sigma * np.sqrt((T-t)) * X[i]) * (g(S_T[i])>0) # again compare formula here
    # Finally determine the hedging position by taking the mean according to formula on page 52
    phi1_t = np.mean(Zeta)

    return phi1_t


# Test the function for given parameters and compare with formula 3.30) from the lecture notes from page 40
t = 0
S0 = 100.00
r = 0.05
sigma = 0.2
T = 1
N = 10000
seed = 33
set_seed = True
# Define the given payoff function g(x)
K = 90.0 # strike K = 90
g = lambda x: np.maximum(0, x-K )
phi1_t = EuOptionHedge_BS_MC_IP (St=S0, r=r, sigma=sigma, g=g, T=T, t=t, N=N, seed=seed, set_seed=True)
# 0.8176244641920389
#Compare the implemented version against the formula 30 from page 40
def phi1_t_comp(St: float, r: float, sigma: float, T: int, t: int, K: float) -> float:
    """Just code the formula 30 from page 40 for deriving the hedging position

    Args:
        St (float): Current underlying (stock) price
        r (float): Interest rate
        sigma (float): Volatility
        T (int): Time Horizon
        t (int): Current time step
        K (float): Strike price of the option

    Returns:
        float: Delta-Hedging position
    """
    # Check for Data type
    for var in [r, sigma, St, K]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, t]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")

    phi1_t_ = norm.cdf((np.log(St / K) + r * (T-t) + ((sigma**2) / 2) * (T-t)) / (sigma * np.sqrt((T-t))), loc=0, scale=1)

    return phi1_t_


phi1_t_formula = phi1_t_comp(St=S0, r=r, sigma=sigma, T=T, t=t, K=K)
# 0.8097030607754923
# Show them side by side
print(f"The hedge with the position phi1_t according to infinitesimal perturbation is: {phi1_t}, whereas the explicit formula acc. to page 40 yields: {phi1_t_formula}")
print(f"Leading to an error of: {phi1_t_formula-phi1_t} (formula-simulation)")

plt.plot(S0, phi1_t,"o", label="Infin. Pert.", color="black")
plt.plot(S0, phi1_t_formula,"o", label="Expl. form", color="grey")
plt.xlabel(f"Initial stock price: {S0}")
plt.ylabel("phi1_t values")
plt.title("Infinitesimal perturbation for determining the hedging position phi1_t")
plt.legend()
plt.show()
