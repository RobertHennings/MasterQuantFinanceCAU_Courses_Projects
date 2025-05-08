# Sheet: 08, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
# C-Exercise 31
# Simulating paths of geometric Brownian motion

def Sim_Paths_GeoBM(X0: float, mu: float, sigma: float, T: int, N: int) -> list:
    """simulates the discrete path approximation via three methods:
        1) exact solution for the geometric Brownian motion SDE
        2) Euler method
        3) Milshtein method

    Args:
        X0 (float): Starting value of the paths
        mu (float): Long Term average of the process
        sigma (float): Drift of the process
        T (int): Time Horizon (1 discretized period)
        N (int): Number of Time Steps for discretization

    Returns:
        list: Simulated process values for:
              1) exact solution for the geometric Brownian motion SDE
              2) Euler method
              3) Milshtein method
    """
    # Check for datatype
    for var in [X0, mu, sigma]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, N]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    # Initialise delta_t and the random normal values
    delta_t = T / N
    # random values for delta_w according to the algo on page no.55
    delta_w = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=N)

    # Set up matrices to save the computed values
    X_exact_mat = np.zeros(N)
    X_euler_mat = np.zeros(N)
    X_milshtein_mat = np.zeros(N)

    # Compute the values in a loop
    # First start the matrices with initial value
    for i in range(0, N-1, 1):
        X_exact_mat[i] = X0
        X_euler_mat[i] = X0
        X_milshtein_mat[i] = X0

    # Compute the values via the three techniques
    for i in range(1, N, 1): # watch out here for start lag
        X_exact_mat[i] = X_exact_mat[i-1] * np.exp((mu - (sigma**2 / 2)) * delta_t + sigma * delta_w[i-1])
        X_euler_mat[i] = X_euler_mat[i-1] * (1 + mu * delta_t + sigma * delta_w[i-1])
        X_milshtein_mat[i] = X_milshtein_mat[i-1] * (1 + mu * delta_t + sigma * delta_w[i-1] + (sigma**2 / 2) * ((delta_w[i-1]**2) - delta_t))
    
    return [X_exact_mat, X_euler_mat, X_milshtein_mat]


# Test the function for given parameters and plot
X0 = 100.0
mu = 0.1
sigma = 0.3
T = 1
N = [10, 100, 1000, 10000]


# For each value of N plot all three simulated paths into a single plot
for ind, n in enumerate(N,start=1):
    plt.subplot(2, 2, ind)
    plt.plot(Sim_Paths_GeoBM(X0=X0, mu=mu, sigma=sigma, T=T, N=n)[0], label="Exact", color="black")
    plt.plot(Sim_Paths_GeoBM(X0=X0, mu=mu, sigma=sigma, T=T, N=n)[1], label="Euler", color="grey")
    plt.plot(Sim_Paths_GeoBM(X0=X0, mu=mu, sigma=sigma, T=T, N=n)[2], label="Milsh", color="#603086")
    plt.ylabel("Path value", fontsize=8)
    plt.xticks(fontsize=6)
    plt.title(f"3 Simulation Methods for N = {n}", fontsize=8)
plt.legend()
plt.show()
