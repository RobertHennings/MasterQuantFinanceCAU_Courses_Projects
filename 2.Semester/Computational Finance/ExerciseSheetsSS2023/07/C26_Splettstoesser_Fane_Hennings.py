# Sheet: 07, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Kadisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
# C-Exercise 26
# Simulate an American Option with payoff defined on Sheet 06 in T20 b)

def CRR_AmOption(S_0: float, T: int, M: int, r: float, sigma: float) -> float:
    # Check for Data type
    for var in [r, sigma, S_0]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T, M]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    # Set r=0 and sigma=sqrt(2) according to task, no strike K involved
    delta_t = (T/M)
    beta = (1/2) * (np.exp(-r*delta_t) + np.exp((r+sigma**2)*delta_t))
    u = beta + np.sqrt(beta**2-1)
    d = (1/u)
    q = (np.exp(r*delta_t) - d) / (u-d)

    stock = np.empty((M+1, M+1))
    payoff = np.empty((M+1, M+1))
    value = np.empty((M+1, M+1))

    for i in range(0, M+1):
        for j in range(0, i+1):
            stock[j, i] = S_0 * u**j * d**(i-j)
            # Here edit the payoff process according to the given function 1) on Sheet 06 T20 b)
            payoff[j, i] = (4 * stock[j, i]**(3/4)) if stock[j, i] < 1 else (3 * np.sqrt(stock[j, i]) + stock[j, i]**(3/2))

    # determine the payoffs at maturity
    value[:, M] = payoff[:, M]

    # Calculate reversed the value
    for i in range(M-1, -1, -1):
        for j in range(0, i+1):
            # American Option style 
            value[j, i] = np.maximum(payoff[j, i], np.exp(-r*delta_t) * (q * value[j+1, i+1] + (1-q) * value[j, i+1]))
    # return stock, payoff, value
    return value[0, 0]

# For comparison also implement the second formula from Sheet 06 T20 b) which directly computes the the option value at any given time t
def CRR_AmOptionDirect(S_0: float, T: int) -> float:
    # Question: How does the stock price evolve? -> Not necessary here to model as general form
    # Set t=0
    t = 0
    # Define the payoff from formula 1) in the first case
    g = lambda S_t: (4 * S_t**(3/4)) if S_t < 1 else (3 * np.sqrt(S_t) + S_t**(3/2))
    # Define the payoff from formula 2) in the other case
    V_1 = lambda t, S_t: np.exp(-(1/4) * (T-t)) * 3 * np.sqrt(S_t) + np.exp((3/4) * (T-t)) * S_t**(3/2)

    # Evaluate based on the condition given in formula 2), S_t = S_0
    V_0 = g(S_0) if S_0 < np.exp(- (T-t))\
        else V_1(t, S_0)

    return V_0



# Set the testing parameters
S_0 = 1.0
T = 1
M = 500
r = 0.0
sigma = np.sqrt(2)

# Print the values and compare
plt.plot(S_0, CRR_AmOption(S_0=S_0, T=T, M=M, r=r, sigma=sigma), "o", color="black", label="CRR Sim")
plt.plot(S_0, CRR_AmOptionDirect(S_0=S_0, T=T), "o", color="grey", label="Direct")
plt.text(x=1.002, y=CRR_AmOption(S_0=S_0, T=T, M=M, r=r, sigma=sigma), s=f"CRR Sim. : {CRR_AmOption(S_0=S_0, T=T, M=M, r=r, sigma=sigma)}")
plt.text(x=1.002, y=CRR_AmOptionDirect(S_0=S_0, T=T), s=f"Direct : {CRR_AmOptionDirect(S_0=S_0, T=T)}")
plt.xlabel(f"Initial Stock Price S_0: {S_0}")
plt.ylabel("Option value")
plt.title("Option Value computed via CRR Sim. vs. Direct Formula")
plt.legend(loc="upper left")
plt.show()

print(f"Difference between modified CRR Simulation:\
      {CRR_AmOption(S_0=S_0, T=T, M=M, r=r, sigma=sigma)} and Direct Compuation:\
      {CRR_AmOptionDirect(S_0=S_0, T=T)} is: {CRR_AmOption(S_0=S_0, T=T, M=M, r=r, sigma=sigma) - CRR_AmOptionDirect(S_0=S_0, T=T)}")
