# Sheet: 03, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings
import scipy
import numpy as np
import matplotlib.pyplot as plt

# C-Exercise 08
def MC_Integration(N: int, f: callable, a: float, b: float) -> float:
    # Take given function and plug in uniformly distributed random numbers
    # Integral as a sum
    scaler = (b-a) / N
    # Evaluate given function and sum up result over the N random samples as x input
    np.random.seed(1)
    U = np.random.uniform(low=a, high=b, size=N)

    single_results = np.zeros(N)
    # plug in given f and sum result
    for ind, u in enumerate(U):
        single_results[ind] = f(u)
    # Compute sum and apply Scale
    approx = scaler * np.sum(single_results)

    return approx


# Test function
N = 10000
a = 0
b = 1
f = lambda x: np.sqrt(1-x**2)


MC_Integration(N, f, a, b)
# 0.7876424120555697

# Test against implemented method
scipy.integrate.quad(f, 0, 1)[0]
# 0.7853981633974481

# Test function with N = 10, 100, 1000, 100000
ns = []
for n in [10, 100, 1000, 10000]:
    ns.append(MC_Integration(n, f, a, b))
    print(f"With a test size of {n} samples we arrive at a MC Integration value of: {MC_Integration(n, f, a, b)}")

# Plot against default integration function
plt.plot([10, 100, 1000, 10000], ns, "-o", color="#603086")
plt.axhline(y=scipy.integrate.quad(f, 0, 1)[0], color="black", label="Scipy Value")
plt.xticks([10, 100, 1000, 10000])
plt.text(x=2500.00, y=0.80, s=f"Scipy package value: {scipy.integrate.quad(f, 0, 1)[0]}")
plt.xlabel("Number of samples drawn: N")
plt.ylabel("MC Integration value")
plt.title("MC Integration over different sample sizes N")
plt.legend()
plt.show()
