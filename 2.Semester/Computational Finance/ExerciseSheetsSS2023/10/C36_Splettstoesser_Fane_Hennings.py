# Sheet: 10, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import scipy.stats

# For Matrix inversion also include the TriagMatrix function from the previous sheet
def TriagMatrix_Inversion(alpha, beta, gamma, b):
    # create the matrix
    n=len(alpha)
    alpha_hat=np.empty(n)
    b_hat= np.empty(n)
    x= np.empty(n)
    # initialize first value
    alpha_hat[0]=alpha[0]
    b_hat[0]=b[0]
    # fill in matrix
    for i in range (1,n):
        alpha_hat[i]= alpha[i]-(gamma[i]/alpha_hat[i-1])*beta[i-1]
        b_hat[i]= b[i]-(gamma[i]/alpha_hat[i-1])*b_hat[i-1]

    x[n-1]= b_hat[n-1]/alpha_hat[n-1]
    for i in range (n-2,-1,-1):
        x[i]= (1/alpha_hat[i])* (b_hat[i]-beta[i]*x[i+1])
    return x

# For later compariosn also define the BSM formula
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (np.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    Call = S_t * scipy.stats.norm.cdf(d_1) - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return Call


# C-Exercise 36
# Valuation of a European Call using the Crank-Nicolsen finite difference scheme
def BS_EuCall_FiDi_CN(r: float, sigma: float, a: float, b: float, m: int, nu_max: float, T: int, K: float) -> float:
    # compute all the static values first as given in the lecture notes
    # see p.65 and p.66 for the algorithm as pseudo code
    q = (2 * r) / (sigma**2)
    delta_x_tilde = (b - a) / m
    delta_t_tilde = ((sigma**2) * T) / (2 * nu_max)
    lambda_ = delta_t_tilde / (delta_x_tilde**2)
    
    # Also compute x_tilde to be able to compute the vector w
    x_tilde = np.arange(a, b, delta_x_tilde)

    # Set up empty matrices to store the values from the loops with m slots each
    w = np.zeros((m, nu_max))
    S = np.zeros(m)
    V0 = np.zeros(m)

    # first compute the values for w, the approximation, choose the Call variant
    call = lambda x, q: np.maximum(np.exp((x / 2) * (q + 1)) - np.exp((x / 2) * (q - 1)), 0)
    put = lambda x, q: np.maximum(np.exp((x / 2) * (q - 1)) - np.exp((x / 2) * (q + 1)), 0)

    for i in range(0, m, 1):
        w[i, 0] = call(x_tilde[i], q)

    # Next we need to set up the triagonalmatrix A for later inversion, use previously defined function
    # k = [(-lambda_ / 2) * np.ones(m-1), (1 + lambda_) * np.ones(m), (-lambda_ / 2) * np.ones(m-1)]
    # offset = [-1,0,1]
    # A = scipy.sparse.diags(k,offset).toarray()
    alpha= np.ones(m - 2) * (1 + lambda_)
    beta = np.ones(m - 2) * (-lambda_ / 2)
    gamma = np.ones(m - 2) * (-lambda_ / 2)

    # Next loop over the nu values
    for i in range (1, nu_max, 1):
        for j in range (1, m-1, 1):
            w[j, i] = lambda_ / 2 * w[j-1, i-1] + (1 - lambda_) * w[j, i-1] + lambda_ / 2 * w[j+1, i-1]
        w[m-2, i] = w[m-2,i] + lambda_ / 2 * (np.exp((q + 1) / 2 * b + (q + 1)** 2 / 4 * i * delta_t_tilde) - np.exp((q - 1) / 2 * b + (q - 1) ** 2 / 4 * i * delta_t_tilde))
        w[1:-1, i] = TriagMatrix_Inversion(alpha, beta, gamma, w[1:-1, i])
        w[m-1, i] = np.exp((q + 1) / 2 * b + (q + 1) ** 2 / 4 * i * delta_t_tilde) - np.exp((q - 1) / 2 * b + (q - 1) ** 2 / 4 * i * delta_t_tilde)
    for i in range (0, m, 1):
        S[i]= K * np.exp(x_tilde[i])
        V0[i]= K * w[i, nu_max-1] * np.exp((-x_tilde[i] / 2) * (q - 1) - sigma**2 / 2 * T * ((q - 1)**2 / 4 + q))
    return [S, V0]


# Set testing parameters
r = 0.05
sigma = 0.2
a = -0.7
b = 0.4
m = 100
nu_max = 2000
T = 1
K = 100.0
# Obtain the value
BS_EUCallFiDi_results = BS_EuCall_FiDi_CN(r, sigma, a, b, m, nu_max, T, K)


# Plot difference of default BSM Formula for all underlying initial stock prices
# Compute the BSM price for every stock price from the set up function
# Store results of the BSM formula
V0_BSM = np.zeros(len(BS_EUCallFiDi_results[0]))
# Compute the price 
for i in range (0,len(BS_EUCallFiDi_results[0]),1):
    V0_BSM[i]=EuCall_BlackScholes(0, BS_EUCallFiDi_results[0][i], r, sigma, T, K)

# Plot the differences
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(BS_EUCallFiDi_results[1], color="#603086", label="FiDi")
ax1.plot(V0_BSM, color="black", label="BSM")
ax1.set_title("EU Call: Finite Difference approach vs. BSM Formula")
ax1.xlabel("Initial Stock prices")
ax1.ylabel("Option prices")
ax1.legend()
ax2.plot(BS_EUCallFiDi_results[1] - V0_BSM, color="grey")
ax2.set_title("Error")
plt.show()
