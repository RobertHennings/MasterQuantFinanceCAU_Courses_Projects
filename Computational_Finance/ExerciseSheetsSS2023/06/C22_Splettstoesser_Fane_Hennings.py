# Sheet: 06, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Karisatou Fane, Robert Hennings
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate as integrate

# C-Exercise 22
# Greeks of a European option in the Black-Scholes model

# Since the bespoken function BS_Price_Int isnt available in the Materials folder as of now (20.05.23, 17:20)
# we code it on our own
# formula 3.21 according to the lecture notes

# a)
# Set the values
# Define Payoff function given in b)
g = lambda x: np.maximum(x-110.0, 0)
r = 0.05
sigma = 0.3
T = 1
S0 = np.arange(60.0, 141.0, 1)
eps = 0.001

def BS_Price_Int(r: float, sigma: float, S0: float, T: int, g: callable) -> float:
    """Compute EU Call style initial value for a given payoff function g (either Call or Put)

    Args:
        r (float): Interest rate
        sigma (float): Volatility
        S0 (float): Initial underlying (stock) price
        T (int): Time Horizon
        g (callable): payoff function (either Call or Put)

    Returns:
        float: Initial Option Value V0
    """
    # Check for Data types
    for var in [r, sigma, S0]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    # we will apply the pricing by integration from lecture page 37 and formula 3.21
    # because we can later use the dedicated integrate function of the scipy package we will just
    # define the function in dependence of x and not mess with any integration by ourself
    # later use: scipy.integrate.quad(function, from_integration, to_integration)

    def Value_BS_EU_Option(x):
        V = (1 / np.sqrt(2 * np.pi)) * g(S0 * np.exp((r - (sigma**2 / 2)) * T + sigma * np.sqrt(T) * x)) * np.exp(-r * T) * np.exp(- (x**2 / 2))
        return V

    # Now use the integration function of the scipy package and put in the borders
    V0 = scipy.integrate.quad(Value_BS_EU_Option, -np.inf, np.inf)
    # save final integration value from -inf up to inf
    V0 = V0[0]

    return V0


# Next write a function that computes the partial derivatives (Greeks) of the BSM formula using approximation
def BS_Greeks_num(r: float, sigma: float, S0: float, T: int, g: callable, eps: float) -> list:
    """Compute the Delta, Gamma and Vega for EU Option style and given payoff function

    Args:
        r (float): Interest rate
        sigma (float): Volatility
        S0 (float): Initial underlying (stock) price
        T (int): Time Horizon
        eps (float): epsilon for approximation method

    Returns:
        list: Delta, Gamma and Vega for a EU Option style and given payoff function 
    """
    for var in [S0, r, sigma, eps]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")
    # Use the predefined function 3.21) to determine the initial value of the option
    V0 = BS_Price_Int(r=r, sigma=sigma, S0=S0, T=T, g=g)
    # Compute the Greeks with approximation from the sheet
    # f(x, y) as the value of the option with x as stock price input
    # Delta as just the initial stock price modified by the eps component - normal computed option value
    # divided by the mdified initial stock price
    Delta = (BS_Price_Int(S0=S0+eps*S0,r=r,sigma=sigma,T=T,g=g) - V0)/(eps*S0)
    # Just as before with the Delta we apply the general approxmation formula here but 
    # now instead of modifying the stock price we will modify the respective variable: sigma
    Vega = (BS_Price_Int(S0=S0,r=r,sigma=sigma+eps*sigma,T=T,g=g)- V0)/(eps*sigma)
    # For the Gamma we need to apply the second given formula from the approximation
    # with modified initial stock price
    Gamma = (BS_Price_Int(S0=S0+eps*S0,r=r,sigma=sigma,T=T,g=g) - 2 * V0 + BS_Price_Int(S0=S0-eps*S0,r=r,sigma=sigma,T=T,g=g))/(eps*S0)**2

    # Delta: sensitivity w.r.t. change in underlying (stock) price
    # Vega: sensitivity w.r.t. change in underlying's (stock) vola
    # Gamma: delta sensitivity w.r.t. change in underlying's (stock) price
    return [Delta, Vega, Gamma]

# b)
# Plot the Delta evolution of a EU Call Option over different intial stock prices S0
Deltas = np.zeros(len(S0))
# Compute the Deltas
for i in range(len(Deltas)):
    Deltas[i] = BS_Greeks_num(r=r, sigma=sigma, S0=S0[i], T=T, g=g, eps=eps)[0]

# plot Delta vs. Initial stock price
plt.plot(S0, Deltas, color="#603086")
plt.axvline(x=110.0, color="black")
plt.text(x=112.0, y=np.mean(Deltas), s="Strike Price: 110")
plt.xlabel("Initial Stock Price S0")
plt.ylabel("EU Call Option Delta Value")
plt.title("EU Call Option BSM Framework Delta vs. Initial Stock Price")
plt.legend()
plt.show()
