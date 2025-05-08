# C-Exercise 31, SS 2023
import numpy as np
import math
import matplotlib.pyplot as plt

def Sim_Paths_GeoBM(X0, mu, sigma, T, N):
    Delta_t = T/N
    Delta_W = np.random.normal(0, math.sqrt(Delta_t), (N,1))

    #Initialize vectors with starting value
    X_exact = X0 * np.ones(N+1)
    X_Euler = X0 * np.ones(N + 1)
    X_Milshtein = X0 * np.ones(N + 1)

    #Recursive simulation according to the algorithms in Section 4.2 using identical Delta_W
    for i in range(0, N):
        X_exact[i+1] = X_exact[i] * np.exp((mu- math.pow(sigma,2)/2)*Delta_t + sigma * Delta_W[i])
        X_Euler[i+1] = X_Euler[i] * (1 + mu * Delta_t + sigma * Delta_W[i])
        X_Milshtein[i+1] = X_Milshtein[i] * (1+mu*Delta_t + sigma*Delta_W[i] + math.pow(sigma,2)/2*(math.pow((Delta_W[i]), 2)- Delta_t))

    return X_exact, X_Euler, X_Milshtein

#test parameters
X0 = 100
mu = 0.1
sigma = 0.3
T = 1
N = np.array([10,100,1000,10000])

#plot
plt.clf()
for i in range(0,4):
    X_exact, X_Euler, X_Milshtein = Sim_Paths_GeoBM(X0, mu, sigma,T, N[i])
    plt.subplot(2,2,i+1)
    plt.plot(np.arange(0, N[i]+1)*T/N[i], X_exact, label = 'Exact Simulation')
    plt.plot(np.arange(0, N[i]+1) * T / N[i], X_Euler, 'red', label = 'Euler approximation')
    plt.plot(np.arange(0, N[i]+1) * T / N[i], X_Milshtein, 'green', label = 'Milshtein approximation')
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.title('N=' + str(N[i]))
    plt.legend()

plt.show()