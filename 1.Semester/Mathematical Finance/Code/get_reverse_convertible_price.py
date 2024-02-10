# Pricing of a reverse convertible bonds
def get_reverse_convertible_bond_price(N: float, K: float, S_1_T: float, T: int, r: float) -> float:
    import numpy as np
    assured_interest = N * (np.exp(r * T) - 1)
    if S_1_T > K:
        S_2_T = N * np.exp(r * T)
        print(f"{S_1_T} > {K} only nominal amount: {N} plus assured interest: {assured_interest} is paid out to buyer of the convertible")
    else:
        S_2_T = (N/K) * S_1_T + assured_interest
        print(f"{S_1_T} < {K} the fraction of shares (N/K): {round(N/K, 2)} has to be provided plus the assured interest: {assured_interest}")
    return S_2_T

N = 100
K = 90
S_1_T = 85
T = 1
r = 0

get_reverse_convertible_bond_price(N=N, K=K, S_1_T=S_1_T, T=T, r=r)

put_option = lambda K, S_1_T: max(K-S_1_T, 0)
put_option(K, S_1_T)
bond = lambda N, r, T: N * np.exp(r * T)

def get_recreated_conv_payoff(option: callable, bond: callable, N: float, K: float, S_1_T: float, T: int, r: float) -> float:
    import numpy as np
    assured_interest = N * (np.exp(r * T) - 1)
    print(f"In t=0 we get as the seller the nominal amount: {N} and invest that into a bond that earns interest")
    print(f"After T={T} the bond has the value: {bond(N, r, T)} with earned interst: {assured_interest}")
    print(f"We also go short a put option (sell a put option)")
    if S_1_T > K:
        print(f"{S_1_T} > {K} so we have to provide the buyer of the convertible the nominal amount: {N} and the assured interst: {assured_interest} back\
              we just give him the amount from the invested bond: {bond(N, r, T)}, the put option expires worthless: {put_option(K, S_1_T)}")
    else:
        print(f"{S_1_T} < {K} so we have to provide the buyer of the convertible the assured interst: {assured_interest} back\
              we just give him the amount from the invested bond, the put option expires: {-put_option(K, S_1_T)} what means we have to\
              buy the underlying asset for the strike K: {K} from the holder of the put option with the rest of the invested bond amount\
              we buy the underlying in the ratio (N/K): {round(N/K, 2)}")
        
get_recreated_conv_payoff(option=put_option, bond=bond, N=N, K=K, S_1_T=S_1_T, T=T, r=r)