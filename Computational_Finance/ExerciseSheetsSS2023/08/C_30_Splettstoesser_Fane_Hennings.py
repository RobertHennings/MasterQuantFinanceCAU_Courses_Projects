# Sheet: 08, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
# C-Exercise 30
# Valuation of European options in the Heston model using the Euler method

def Heston_EuCall_MC_Euler(S0: float, r: float, gamma_0: float, kappa: float,
                           lambda_: float, sigma: float, T: int, g: callable,
                           M: int, m: int) -> list:
    """_summary_

    Args:
        S0 (float): Starting value for the simulated underlying (stock)
                    processes
        r (float): Interest rate
        gamma_0 (float): Starting value for the simulated volatility
                         processes
        kappa (float): _description_
        lambda_ (float): _description_
        sigma (float): Volatility of the simulated Volatility processes
        T (int): Time horizon
        g (callable): Payoff function (Call or Put type)
        M (int): Monte-Carlo method using M samples
        m (int): discretization of time steps, grid size

    Returns:
        list: discounted initial option price acc. to the Heston model,
              lower and upper confidence interval for option price
    """
    # Check data types
    for var in [S0, r, gamma_0, kappa, lambda_, sigma]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, M, m]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    # approximate the paths by the Euler method, see the defined algorithm on page 54, 55 in the script
    # Recall the Heston model assumes time variant volatility
    # Initalize the time delta steps
    delta_t = T / m # watch out small M here

    # Initialize arrays to store option prices and paths, length M of
    # number of Monte-Carlo samples
    St_mat = np.zeros((M, m))
    gammat_mat = np.zeros((M, m))
    # Brownian motions for eq. 3.37 and eq.3.39
    delta_w_1 = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=(M, m))
    delta_w_2 = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=(M, m))
    # Idea: we want to use the Heston model and therefore its nature with
    # stochastic processes for the underlying as well as for the volatility
    # we want to simulate the price and vola processes and plug in the Heston model
    # Set the starting values given
    St_mat[:, 0] = S0 * np.ones(M)
    gammat_mat[:, 0] = gamma_0 * np.ones(M)

    for i in range(1, m):
        # compute the gamma process following page 46, eq. 3.37
        # γ(t) = σ2(t) is assumed to satisfy the SDE
        gam = gammat_mat[:, i-1] + (kappa - lambda_ * gammat_mat[:, i-1]) * delta_t + sigma * np.sqrt(gammat_mat[:, i-1]) * delta_w_1[:, i-1]
        gammat_mat[:, i] = np.maximum(gam, 0)
        # now use the gamma process in the Heston model for the underlying (stock)
        # following page 46, eq. 3.39 in undiscounted terms directly
        St_mat[:, i] = St_mat[:, i-1] + r * St_mat[:, i-1] * delta_t + St_mat[:, i-1] * np.sqrt(np.maximum(gammat_mat[:, i-1], 0)) * delta_w_2[:, i-1]
    # after the simulation is completed and we have for each i in m simulations M
    # we compute the option payoff at the terminal time point and discount
    # back to starting time point
    raw_undiscounted_payoff = g(St_mat[:, -1])
    # now discount back to t0
    V0 = np.exp(-r * T) * np.mean(raw_undiscounted_payoff)
    # next compute the confidence intervals
    std_err = np.std(raw_undiscounted_payoff) / np.sqrt(M)
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
g = lambda x: np.maximum(x - K, 0)
M = 10000
m = 250

V0, c1, c2 = Heston_EuCall_MC_Euler(S0=S0, r=r, gamma_0=gamma_0, kappa=kappa,
                                    lambda_=lambda_, sigma=sigma, T=T, g=g,
                                    M=M, m=m)
print(f"Calculated option price using the Heston model (Call): {V0}, with lower 95% CI:{c1}, and upper 95% CI:{c2}")

# Visualize the convergence as the number of Monte-Carlo Simulations rises over time
mc_numbers = [int(num) for num in np.arange(start=100, stop=10000, step=500)]
v0_list = []
c1_list = []
c2_list = []

for mc_n in mc_numbers:
    V0, c1, c2 = Heston_EuCall_MC_Euler(S0=S0, r=r, gamma_0=gamma_0, kappa=kappa,
                                    lambda_=lambda_, sigma=sigma, T=T, g=g,
                                    M=mc_n, m=m)
    v0_list.append(V0)
    c1_list.append(c1)
    c2_list.append(c2)

# plot the convergence
plt.plot(mc_numbers, v0_list, label="V0", color="#603086")
plt.plot(mc_numbers, c1_list, label="c1", color="grey")
plt.plot(mc_numbers, c2_list, label="c2", color="black")
plt.xlabel("Number of MC-Simulations")
plt.ylabel(r"Option price, lower and upper 95% confidene interval")
plt.title("Convergence over time: Heston Model Option prices", fontsize=12)
plt.legend()
plt.show()
