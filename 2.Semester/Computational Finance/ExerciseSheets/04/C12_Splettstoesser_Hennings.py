# Sheet: 04, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st


# C-Exercise 12
# (Pricing a deep out-of-the-money European call option by Monte-Carlo with importance sampling
S0 = 100
r = 0.05
sigma = 0.3
K = 220
T = 1
N = 10000
alpha = 0.95
mu = 0.4

t = 0
# Define the function
def BS_EuCall_MC_IS(S0: float, r: float, sigma: float, K: float, T: int, mu: float, N: int, alpha: float) -> tuple:
    """Compute the European Call option price using Monte Carlo simulated stock prices and the
       respective confidence interval of the simulated Option values but this time applying Importance Sampling technique

    Args:
        S0 (float): Initial stock price
        r (float): Interest rate
        sigma (float): Volatility
        T (int): Time horizon
        K (float): Strike price
        M (int): Discretization of the Time horizon
        f (callable): Payoff function, either Call or Put type

    Returns:
        tuple: the approximated price value_is obtained via importance (MC) sampling
               and the respective confidence interval boundaries of level alpha
    """

    # Use a new random variable Y ~ N(mu, 1) for the importance sampling
    # Orientation provides script: page 22
    # Idea: use the exercise C10 from Sheet 03 and modify the sampling from X ~ N(0,1) to Y ~ N(mu, 1)

    # np.random.seed(22)
    # Instead of X ~ N(0,1) we now simulate Y ~ N(mu, 1) to change our sample results
    # and shift our distribution in the according direction of mu more up or down
    Y = np.random.normal(loc=mu, scale=1, size=N)

    # Define the Call function from C10 from sheet 03
    def f(x, K):
        return np.maximum(x-K, 0)
    # Compute terminal stock price based on simulated Ys, apply slight modifications
    S_T = np.zeros(N)
    payoffs = np.zeros(N)
    value = np.zeros(N)

    for i in range(len(Y)):
        S_T[i] = S0 * np.exp((r - sigma**2/2) * T + sigma * np.sqrt(T) * Y[i])
        payoffs[i] = f(x=S_T[i], K=K)
        # Apply some changes here since we have edited the sampling distribution
        # value[i] = np.exp(-r * T - Y[i] * mu + (mu**2 / 2)) * payoffs[i]
        value[i] = np.exp(-r * T - Y[i] * mu + (mu**2 / 2)) * payoffs[i]

    # also get the confidence interval here
    quantiles = st.t.interval(alpha=alpha, df=len(value)-1, loc=np.mean(value), scale=st.sem(value))
    # As we take from the lecture notes f.2.1) take here the emprical mean for the expected value
    value_is = np.mean(value)
    return value_is, quantiles[0], quantiles[1]


# Test the function
BS_EuCall_MC_IS(S0, r, sigma, K, T, mu, N, alpha)


# Compare with the BSM price from Exercise Sheet 02, EU Call
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    # Compute static values first
    # Compute d1 and d2
    d1 = (np.log(S_t/K) + (((sigma**2)/2)+r) * (T-t)) / (sigma * (np.sqrt(T-t)))
    d2 = d1 - sigma * (np.sqrt(T-t))

    EuCall = S_t * norm.cdf(d1) - K * (np.exp(-r * (T-t)) * norm.cdf(d2))

    return EuCall

BSM_CallEU_P = BlackScholes_EuCall(t=t, S_t=S0, r=r, sigma=sigma, T=T, K=K)


# Plot the simulation results against the true price
mu = np.linspace(-8.0, 8.0, 100)
V0_hat = np.zeros(len(mu))

for ind, mu_ in enumerate(mu):
    V0_hat[ind] = BS_EuCall_MC_IS(S0, r, sigma, K, T, mu_, N, alpha)[0]

# plot the results
plt.plot(mu, V0_hat, color="#603086", label="Sim.")
plt.axhline(y=BSM_CallEU_P, color="black", label="Act.")
plt.text(x=min(mu)+2, y=BSM_CallEU_P+0.02, s=f"BSM price: {round(BSM_CallEU_P, 4)}")
plt.title("EU Call option price via MC IS method vs. BSM fair price")
plt.ylabel("EU Call Option price")
plt.xlabel("mu for MC IS sampling")
plt.legend()
plt.show()


# Is there an optimal mu to choose?
# See script page 22 where all values Xn > d contribute to the empirical mean
# apply such a range for mu

d = (np.log(K / S0) - (r - (sigma**2 / 2)) * T) / sigma * np.sqrt(T)
mu = np.linspace(0, d+3, 400)
V0_hat = np.zeros(len(mu))


# Check results again
for ind, mu_ in enumerate(mu):
    V0_hat[ind] = BS_EuCall_MC_IS(S0, r, sigma, K, T, mu_, N, alpha)[0]

# plot the results
plt.plot(mu, V0_hat, color="#603086", label="Sim.")
plt.axhline(y=BSM_CallEU_P, color="black", label="Act.")
plt.text(x=min(mu)+2, y=BSM_CallEU_P+0.02, s=f"BSM price: {round(BSM_CallEU_P, 4)}")
plt.title("EU Call option price via MC IS method vs. BSM fair price")
plt.ylabel("EU Call Option price")
plt.xlabel("mu for MC IS sampling")
plt.legend()
plt.show()
