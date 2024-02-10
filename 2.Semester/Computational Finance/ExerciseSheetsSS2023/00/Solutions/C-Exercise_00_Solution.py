# C-Exercise 00, SS 2023

import math


def bond_value(V0, r, n, c, M):
    # continuous rate
    if (c == 1):
        Vn = V0 * math.exp(r * n)
        return Vn

    # simple rate
    elif (c == 0):
        Vn = V0 * math.pow((1 + (r / M)), n * M)
        return Vn

    # wrong value for parameter c
    else:
        print('Error: Argument for compound type must be 1 (continuous) or 0 (simple).')


# test parameters
V0 = 1000
r = 0.05
n = 10
c = 0
M = 4

print(bond_value(V0, r, n, c, M))
