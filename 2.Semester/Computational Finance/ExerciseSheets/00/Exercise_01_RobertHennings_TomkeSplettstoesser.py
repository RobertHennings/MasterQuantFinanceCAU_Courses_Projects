# Sheet: 00, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings

# Name: Tomke Splettstoesser
# Student ID: 

# Name: Robert Hennings
# Student ID: 

import math


def bond_value(V0: float, r: float, n: int, M: int, c: int) -> float:
    """ function that calculates terminal value of interest payment
        on an initial provided value

    Args:
        V0 (float): initial starting wealth > 0
        r (float): interest rate > 0
        n (int): number of periods that interest is paid
        M (int): subperiods interest is paid per year
        c (int): 0 for simple rate payment, 1 for continous compounding

    Returns:
        V_n (float): terminal wealth after n periods paid interest 
    """

    # check if numbers are passed correct
    for var in [V0, r]:
        if not isinstance(var, float):
            raise TypeError(f"{var} not type float")
    for var in [n, M, c]:
        if not isinstance(var, int):
            raise TypeError(f"{var} not type int")

    if V0 <= 0:
        print("Requirement V0 > 0 not satisfied: ", V0)

    if r <= 0:
        print("Requirement r > 0 not satisfied: ", r)

    if c == 1:
        # compute in a continous way
        V_n = V0*math.exp(r*n)

    elif c == 0:
        # compute simple rate paid over M time periods per year
        V_n = V0 * (1 + r/M)**(n*M)

    else:
        print("Provide correct vaue for c: either 0 or 1")

    try:
        return round(V_n, 3)
    except:
        print("Wrong parameter inputs, not able to calculate V_n")

# test the function


bond_value(V0=1000.00, r=0.05, n=10, M=4, c=0)
