# C-Exercise 09, SS 2023

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss


# pdf of truncated exponential distribution
def density(x, alpha, beta):
    if 0 <= x and x <= 1:
        factor = ss.gamma(alpha+beta)/(ss.gamma(alpha)*ss.gamma(beta))
        return factor* math.pow(x, alpha-1) * math.pow(1-x, beta-1)
    else:
        return 0


def Sample_dist_AR(N, alpha, beta):
    # compute C as the maximum of f(x)/g(x), such that f(x) <= C*g(x)
    C = density(0.2, alpha, beta)+1

    # function to generate a single sample of the distribution
    def SingleSample():
        # set run parameter for while-loop
        success = False
        while ~success:
            # generate two U([0,1]) random variables
            U = np.random.uniform(size=(2, 1))
            # scale one of them to the correct interval
            Y = U[0]
            # check for rejection/acceptance
            # when the sample gets rejected the while loop will generate a new sample and will check again
            success = ((C * U[1]) <= density(Y, alpha, beta))
        return Y

    # use function SingleSample N times to generate N samples
    X = np.empty(N, dtype=float)
    for i in range(0, N):
        X[i] = SingleSample()

    return X


# test parameters
N = 10000
alpha = 2
beta = 5

X = Sample_dist_AR(N, alpha, beta)
# plot histogram
plt.hist(X, 50, density=True)

# plot exact pdf
x = np.linspace(0 - 1/20, 1 + 1 / 20, 1000)
pdf_vector = np.zeros(len(x))
for i in range(0, len(x)):
    pdf_vector[i] = density(x[i], alpha, beta)
plt.plot(x, pdf_vector)
plt.show()
