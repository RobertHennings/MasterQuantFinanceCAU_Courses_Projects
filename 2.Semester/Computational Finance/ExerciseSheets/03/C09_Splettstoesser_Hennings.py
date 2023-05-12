# Sheet: 03, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
# C-Exercise 09
# Generate samples deom the beta distribution by the acceptance/rejection method

# first define f(x) the density of the beta function
def f_beta(alpha: float, beta: float, x: any) -> float:
    res = (gamma(alpha+beta) / (gamma(alpha) * gamma(beta))) * x**(alpha-1) * (1-x)**(beta-1)
    return res


# Plot the function
x = np.linspace(0, 1, 400)
y = []
for x_ in x:
    y.append(f_beta(alpha=2, beta=5, x=x_))


plt.plot(x, y, color="#603086")
plt.title("Beta distribution(a=2, b=5)")
plt.axhline(y=1, color="black")
plt.text(x=0.1, y=1.4, s="Possib. c values")
plt.fill_between(x=x, y1=1,y2=y, where=(x<0.490) & (x>0.041), color="grey", alpha=0.6)
plt.text(x=0.6, y=1.08, s="Cutoff Uniform for c")
plt.xlabel("X")
plt.ylabel("Beta PDF")
plt.show()


# Next write the sampler
# Draw two random uniformly distributed variables and plug in and compare by criterion

# From the plot of the density we can see that the maximizing y value is about 2.5
# a guess therefore for the parameter c could be the mode (most often occuring value)
# thats about x=0.2 whats in line with the plot
mode = (2-1)/(2+5-2) 
c = f_beta(alpha=2, beta=5, x=mode)  # get maximizing value


def Sample_dist_AR(N: int, alpha: float, beta: float) -> np.array:
    """Compute Samples from a Beta distribution via the Acceptance, rejection method 

    Args:
        N (int): Number of sample that should be drawn from the desired distribution
        alpha (float): alpha param of Beta distribution
        beta (float): beta param of Beta distribution

    Returns:
        np.array: array of N samples
    """
    np.random.seed(1)
    result = np.empty(N)
    counter = 0

    while counter < N:
        # Generate first two random uniformly distributed variables
        # on interval [0, 1]
        U1 = np.random.uniform(low=0, high=1)
        U2 = np.random.uniform(low=0, high=1)
        # Next apply the acceptance/rejecion criterion on what values to keep
        # since we use the uniform distribution as comparison as g(x) we just assume its 1
        c = 2.457
        if U1 <= (1/c) * f_beta(alpha=alpha, beta=beta, x=U2):
            result[counter] = U2
            counter += 1
    return result

# Next generate N = 10000 samples and compare them in a historgram to the real Beta distribution
sample = Sample_dist_AR(N=10000, alpha=2, beta=5)
x_beta = np.linspace(0, 1, len(sample))
y_beta = f_beta(2, 5, x_beta)


plt.hist(sample, density=True, color="grey", bins=60, label="Sim")
plt.plot(x_beta, y_beta, "-", color="#603086", label="Act.")
plt.xlabel("X")
plt.ylabel("Density")
plt.title("Simulated Density of Beta(a=2, b=5) distribution by the Acceptance/Rejection Method", fontsize=10)
plt.legend()
plt.show()
