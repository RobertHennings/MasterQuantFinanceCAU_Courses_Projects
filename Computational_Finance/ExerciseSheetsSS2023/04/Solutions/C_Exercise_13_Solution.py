# C-Exercise 13, SS 2023

import math
import numpy as np
import scipy.stats


#function for computing the price of a self-quanto call in the BS-model using Monte-Carlo with control variate
def EuOption_BS_MC_CV(S0, r, sigma, T, K, M):
    #generate first sample
    X = np.random.normal(0,1, M)
    #compute stock
    ST = S0* np.exp((r-0.5*math.pow(sigma,2))*T + sigma*math.sqrt(T)*X)
    #compute payoff of self-quanto call
    VT = ST*np.maximum(ST-K,0)
    #compute payoff of european call
    CT = np.maximum(ST-K, 0)
    C0 = EuCall_BlackScholes(0, S0, r, sigma, T, K)

    #alternative way to compute beta
    beta_alt = (np.cov(VT, CT)/np.var(CT))[0][1]
    print('Alternative beta is:' + str(beta_alt))
    #compute numerator for beta first
    #C0[0]*math.exp(r*T) is the undiscounted value of the call = E[max(ST-K,0)]
    Covar = np.mean(np.multiply((CT-C0[0]*math.exp(r*T)), (VT- np.mean(VT))))
    #compute beta
    beta = Covar/np.var(CT)
    print('beta is:' + str(beta))

    #generate second sample
    X = np.random.normal(0,1,M)
    #compute stock
    ST_hat = S0* np.exp((r-0.5*math.pow(sigma,2))*T + sigma*math.sqrt(T)*X)
    #compute Monte-Carlo estimator using the initial call option as control variate
    Y = ST_hat * np.maximum((ST_hat-K),0)- beta*np.maximum((ST_hat-K),0)
    V0 = math.exp(-r*T)*np.mean(Y)+ beta*C0[0]
    epsilon = 1.96 * math.sqrt(np.var(Y)/M)
    return V0, epsilon, beta



def Eu_Option_BS_MC(S0, r, sigma, T, K, M, f):
    # generate M samples
    X = np.random.normal(0, 1, M)

    # compute ST and Y for each sample
    ST = S0 * np.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * X)
    Y = f(ST, K)

    # calculate V0
    VN_hat = np.mean(Y)
    V0 = math.exp(-r * T) * VN_hat

    # compute confidence interval
    epsilon = 1.96 * math.sqrt(np.var(Y) / M)
    c1 = (VN_hat - epsilon) * math.exp(-r * T)
    c2 = (VN_hat + epsilon) * math.exp(-r * T)
    return V0, c1, c2, epsilon


# computes the price of a call in the Black-Scholes model
def EuCall_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(d_1)
    C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    return C, phi


def g(x, K):
    return np.maximum(x - K, 0)*x

def Y(x,K):
    return np.maximum(x-K,0)


S0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 100000

V0, epsilon, beta = EuOption_BS_MC_CV(S0, r, sigma, T, K ,M)
print('Price of European call by use of Monte-Carlo simulation with control variate: ' + str(V0) + ', radius of 95% confidence interval: '+ str(epsilon) + ' Beta chosen in the estimation procedure: ' + str(beta))




V0, c1, c2, epsilon = Eu_Option_BS_MC(S0, r, sigma, T, K, M, g)
print('Price of European call by use of plain Monte-Carlo simulation: ' + str(V0) + ', radius of 95% confidence interval: ' + str(epsilon))