# C-Exercise 29, SS 2023

import numpy as np
import scipy.stats
import math
import scipy.misc

def EuOptionHedge_BS_MC_IP (St, r, sigma, g, T, t, N):
    h = 0.00000001

    ### calculate derivative with 'scipy.misc.derivative' of the function g(x) at the point S(T). Note that S(T) changes for each simulation.
    def derivative(x):
        return scipy.misc.derivative(g,St * np.exp((r-sigma**2/2)*(T-t)+sigma*np.sqrt(T-t)*x))

    ### alternative way to calculate the derivative with finite differences
    def derivative_alt(x):
        ST = St * np.exp((r-sigma**2/2)*(T-t)+sigma*np.sqrt(T-t)*x)
        return (g((1+h/2)*ST)-g((1-h/2)*ST))/ ( h*ST)


    ### Simulate normal random variables for the simulation of S(T)
    X = np.random.normal(0,1,N)

    ### allocate empty memory space
    delta = np.zeros(N)
    delta2 = np.zeros(N)


    for i in range(0,N):

        ### infinitesimal perturbation approach
        delta[i] = np.exp(-(np.power(sigma,2)/2)*(T-t)+sigma*np.sqrt(T-t)*X[i]) * derivative(X[i])

        ### Bonus(not part of the exercise): Calculate the hedge with finite differences
        ### Note the difference in how the derivative is calculated
        ST_plus = (St+h/2) * np.exp((r - sigma ** 2 / 2) * (T - t) + sigma * np.sqrt(T - t) * X[i])
        ST_minus = (St - h / 2) * np.exp((r - sigma ** 2 / 2) * (T - t) + sigma * np.sqrt(T - t) * X[i])
        delta2[i] = np.exp(-r*(T-t)) * (g(ST_plus)-g(ST_minus))/ ( h)

    return (np.mean(delta),np.mean(delta2))

St = 100
r = 0.05
sigma = 0.2
T = 1
N = 10000
t = 0

def g(x):
    return np.maximum(x-90,0)

(pertub, findiff) = EuOptionHedge_BS_MC_IP (St, r, sigma, g, T, t, N)

print('Hedge with infinitesimal perturbation:'+str(pertub)+ '\nHedge with finite difference approach:'+str(findiff))

#Hedge according to p.48 in lecture notes
def EuCallHedge_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    return scipy.stats.norm.cdf(d_1)

print('Hedge using the BS-Formula:'  + str(EuCallHedge_BlackScholes(t, St, r, sigma, T, 100)))