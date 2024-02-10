import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
"""
Description:
In what follows is the simple implementation of the variance optimal hedging approach
by using calibration via a linnear regression model from already priced options in
a liquidly traded market following the ideas presented on pages 118-124. This script
is used for soling the Sheet 11.
"""

# Main Input components:
# 1) the strike prices K as x values
# 2) the market prices P as y values

K = [k for k in range(100, 106)]
P = [7.453, 6.970, 6.448, 5.958, 5.467, 5.070]

def main(K: list, P: list):
    linear_regression = LinearRegression()
    # Reshape the data for the correct dimensions
    K = np.array(K).reshape(-1, 1)
    P = np.array(P).reshape(-1, 1)
    # fit the linear regression model
    lin_model = linear_regression.fit(K, P)
    # get model coefficients
    beta0 = lin_model.intercept_
    beta1 = lin_model.coef_
    print(f"Model intercept: {beta0}")
    print(f"Model Coefficients: {beta1}")

    # Use the fitted model to interpolate the price for a K=103.5 strike EU Call option
    K_pred=103.5
    P_pred = lin_model.predict(np.array([K_pred]).reshape(-1, 1))
    print(f"For a given strike K: {K_pred} the price according to the fitted linear regression model is: {P_pred} ")


    plt.plot(K.reshape(1, -1).tolist()[0],
            lin_model.predict(K).reshape(1,-1).tolist()[0],
            "-", color="red")
    for k, p in zip(K.reshape(1, -1).tolist()[0], lin_model.predict(K).reshape(1,-1).tolist()[0]):
        plt.annotate(f"{round(p, 3)}€", (k, p))
    plt.title(f"Price-Strike Relationship using linear regression for\n option pricing Calibration, ß0: {beta0}, ß1: {beta1}")
    plt.ylabel("Option Price P")
    plt.xlabel("Strike K")
    plt.scatter(K.reshape(1, -1), P.reshape(1, -1))
    plt.show()
    # For what arbitraty strike K can the model and the market be free of arbitrage?
    # Check for x-axis crossing
    K_xlim = round(-beta0[0] / beta1[0][0], 3)
    print(f"For a strike of K: {K_xlim} we receive an initial option price of 0")

    K_XLIM = np.array([k for k in range(100, round(K_xlim+20), 5)]).reshape(-1, 1)
    K_XLIM_pred = lin_model.predict(K_XLIM).reshape(1, -1)

    plt.plot(K_XLIM.reshape(1, -1).tolist()[0],
            K_XLIM_pred.tolist()[0],
            "-", color="red")
    for k, p in zip(K_XLIM, K_XLIM_pred.tolist()[0]):
        plt.annotate(f"{round(p, 3)}€", (k+1.5, p))
    plt.hlines(y=0, color="black", xmin=min(K_XLIM)[0], xmax=max(K_XLIM)[0])
    plt.scatter(K_XLIM.reshape(1, -1).tolist()[0], K_XLIM_pred)
    plt.annotate(f"K_xlim: {K_xlim}", (110, -5))
    plt.title(f"Price-Strike Relationship using linear regression for\n option pricing Calibration, ß0: {beta0}, ß1: {beta1}")
    plt.ylabel("Option Price P")
    plt.xlabel("Strike K")
    plt.show()

if __name__ == '__main__':
    main(K=K, P=P)
