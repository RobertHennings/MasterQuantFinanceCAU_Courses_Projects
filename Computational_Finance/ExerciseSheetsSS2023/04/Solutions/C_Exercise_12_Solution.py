# C-Exercise 12, SS 2023

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def BS_EuCall_MC_IS(S0, r, sigma, T, K, N, mu, alpha):
    # define payoff function for european call
    def f(x, K):
        return np.maximum(x - K, 0)

    # generate N random variables with N(mu, 1)-distribution
    Y = np.random.normal(mu, 1, N)
    # simulate stock prices and exponential term for the formula on p.22 in the lecture notes
    # we need to use the exponential-function from the numpy package, because Y and therefore ST and EXP are arrays
    ST = S0 * np.exp((r - math.pow(sigma, 2) / 2) * T + sigma * math.sqrt(T) * Y)
    EXP = np.exp(- Y * mu + math.pow(mu, 2) / 2)
    # Compute payoff
    fST = f(ST, K)
    samplesPayoff = EXP * fST

    VN_hat = np.mean(samplesPayoff)
    # compute the Monte-Carlo estimator
    V_IS = math.exp(-r * T) * VN_hat

    # determine asymptotic alpha-level confidence interval based on the sample variance of the Monto-Carlo samples
    sigma_hat = math.exp(-r * T) * np.std(samplesPayoff)
    epsilon = scipy.stats.norm.ppf(1 - (1 - alpha) / 2, 0, 1) * sigma_hat / math.sqrt(N)

    # compute upper and lower boundary
    CIl = VN_hat - epsilon
    CIr = VN_hat + epsilon
    return V_IS, VN_hat, CIl, CIr, epsilon


def Eu_Option_BS_MC(S0, r, sigma, T, K, M, f):
    # generate M samples
    X = np.random.normal(0, 1, M)
    ST = np.empty(len(X), dtype=float)
    Y = np.empty(len(X), dtype=float)

    # compute ST and Y for each sample
    for i in range(0, len(X)):
        ST[i] = S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * X[i])
        Y[i] = f(ST[i], K)

    # calculate V0
    VN_hat = np.mean(Y)
    # calculate V0
    V0 = math.exp(-r * T) * VN_hat

    # compute confidence interval
    epsilon = 1.96 * math.sqrt(np.var(Y) / M)
    c1 = VN_hat - epsilon
    c2 = VN_hat + epsilon
    return V0, VN_hat, c1, c2, epsilon


# computes the price of a call in the Black-Scholes model
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C, phi


# test parameter
S0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 220
N = 10000
alpha = 0.95

# compute optimal mu
d = (math.log(K / S0) - ((r - (1 / 2) * math.pow(sigma, 2)) * T)) / (sigma * math.sqrt(T))
delta = abs(d)
# define range for mu
mu = np.arange(0, 2 * delta, 0.01)


def g(x, K):
    return np.maximum(x - K, 0)


# call function for mu = d and print the result
V0, VN_hat, c1, c2, epsilon = BS_EuCall_MC_IS(S0, r, sigma, T, K, N, d, alpha)
V0_plain, VN_hat_plain, c1_plain, c2_plain, epsilon_plain = Eu_Option_BS_MC(S0, r, sigma, T, K, N, g)
V0_BS = EuCall_BlackScholes(0, S0, r, sigma, T, K)[0]
print(
    "The Monte-Carlo approximation with importance sampling to the price of the European Call option is given by " + str(
        V0) + ";   radius of 95% confidence interval: " + str(epsilon))
print("The Monte-Carlo approximation without variance reduction is given by " + str(
    V0_plain) + ";   radius of 95% confidence interval: " + str(epsilon_plain))
print("The real option price calculated with the BS-Formula is " + str(V0_BS))

# create approximations with different mu
M = len(mu)
V0_mu = np.empty(M)
VN_mu = np.empty(M)
epsilon = np.empty(M)
for i in range(0, M):
    V0_mu[i], VN_mu, c1, c2, epsilon[i] = BS_EuCall_MC_IS(S0, r, sigma, T, K, N, mu[i], alpha)

# plot of standard estimator and IS estimators for mu against the BS price and the resulting confidence interval
plt.clf()
plt.plot(mu, V0_mu, label='Importance sampling estimation')
plt.plot(mu, V0_plain * np.ones(len(V0_mu)), label='Standard estimator')
plt.plot(mu, V0_BS * np.ones(len(V0_mu)), label='Black-Scholes formula')
plt.plot(mu, V0_BS * np.ones(len(V0_mu)) + epsilon, label='Confidence Interval upper')
plt.plot(mu, V0_BS * np.ones(len(V0_mu)) - epsilon, label='Confidence Interval lower')
h = plt.gca()
h.tight_limits = ["on", "on"]

plt.title("Comparison of standard and importance sampling estimators for a European call option")
plt.xlabel("$\mu$")
plt.ylabel("option price")
plt.legend()

plt.show()
