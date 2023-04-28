# Sheet: 01, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings

import math
import datetime as dt
import numpy as np

def CRR_stock(S_0: float, r: float, sigma: float, T: int, M: float) -> np.array:
    """Computes the stock price evolution in the binomial Cox-Ross-Rubinstein model

    Args:
        S_0 (float): the initial stock price from where to start the binomial tree, S(0)>0
        r (float): interest rate, r>0
        sigma(float): volatility, sigma>0
        T (int): max time horizon that needs to be divided into small incerements as time steps
        M (float): number of increments that T should be divided into
                   Recall: Idea is to approximate continous time evolution by taking small increments
    Returns:
        stock_matrix_2col (pd.DataFrame): 2 column dataframe that holds the specific stock movement and its price
        stock_matrix (pd.DataFrame): dataframe that holds the same data but displayed more tree like
    """
    # Check if numbers were passed correct
    for var in [S_0, r, sigma, M]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [T]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")

    # Compute all the static variables first
    # Delta t
    delta_t = T/M
    # beta
    beta = (1/2)*(math.exp(-r*delta_t)+math.exp((r+math.pow(sigma, 2))*delta_t))
    # d and u that control upward and donward scalar
    d = beta - math.sqrt(math.pow(beta, 2)-1)
    u = beta + math.sqrt(math.pow(beta, 2)-1)

    # Create matrix hat will be filled
    stock_price_mat = np.zeros((M+1, M+1))

    # to be able to calculate how long the process took
    start_time = dt.datetime.now()
    # Compute the stock price matrix for every ith step with j upward and i-j downward moves
    for i in range(0, M+1):
        for j in range(i + 1):  # in every time step there are j possible upward moves
            # counter += 1
            # Apply sanity check 1) here: Display the stock price version
            print(f"S_{j}_{i}")
            # Apply sanity check 2) here: see if condition is met j ups and (i-j) must sum up to i moves max
            print("Check: ", j, "+", (i-j), "=", i, "?", j + (i-j) == i)
            # Compute actual value for each step, keep i fix and vary j ups for every possibility
            print(f"S_{j}_{i}", "= S(0)*u^", j, "*d^", i-j, "=", S_0*math.pow(u, j)*math.pow(d, i-j))
            # Save the actual computed value accordingly in the matrix
            stock_price_mat[i, j] = S_0*math.pow(u, j)*math.pow(d, i-j)
    # Calculate how long the process took
    end_time = dt.datetime.now()
    print("Process took: ", (end_time-start_time).seconds, "seconds")

    return stock_price_mat

CRR_stock(S_0=100, r=0.05, sigma=0.3, T=1, M=500)

S = CRR_stock(S_0=100, r=0.05, sigma=0.3, T=1, M=500)
