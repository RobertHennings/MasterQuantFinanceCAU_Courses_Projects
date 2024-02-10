def get_combinations(n: int, k: int) -> int:
    import pandas as pd
    comb_table = pd.DataFrame(data="", columns=["With replacement", "Without replacement"], index=["Order matters", "Order doesn't matter"])
    def factorials(n: int):
        fac = []; fac = [fac[-1] for i in range(n+1) if not fac.append(i*fac[-1] if fac else 1)]
        return fac[-1]
    comb_table.loc["Order matters", "With replacement"] = n**k
    comb_table.loc["Order matters", "Without replacement"] = (factorials(n) / factorials((n-k)))
    comb_table.loc["Order doesn't matter", "With replacement"] = (factorials((n+k-1)) / (factorials(k) * factorials((n-1))))
    comb_table.loc["Order doesn't matter", "Without replacement"] = (factorials(n) / (factorials(k) * factorials((n-k))))
    return comb_table

n=3
k=3

get_combinations(n=n, k=k)
