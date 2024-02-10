import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy

"""
Description:
In what follows is the implementation of the simulation via stochastic Integral
form based on the modelling approaches of the course Mathematical Finance.
The stochastic integral follows formula 4.3 on page 62 with the increments
following formula 4.4 on page 62.
Also considered is the inverse gaussian sampling and a complex called form,
pages 63 - 66, where a dynamic volatility process is added to the sampling.
The inverse gaussian approach puts more weight on the tails and the centre.
The complex form accounts for the phenomen of volatility clustering by adding
a separated volatilty process multiplicative to the samples.
"""

def main():
    number_of_assets = 5
    len_process = 2000
    sd_process = 0.02
    mean_process = 0.0005
    process_law = "complex" # gaussian, inv_gauss, complex
    S_0_1 = 2000
    DAX_yf_ticker = "^GDAXI"

    def generate_stochastic_exponential_process(number_of_assets: int, len_process: int,
                                                sd_process: float, mean_process: float,
                                                S_0_1: float, process_law: str) -> pd.DataFrame:
        """Generates stochastic exponential form for simulation of asset price
           evolution over time.

        Args:
            number_of_assets (int): number of asset runs that should be simulated 
            len_process (int): discrete time periods
            sd_process (float): sd for simulation of a certain provided law
            mean_process (float): mean for simulation of a certain provided law
            S_0_1 (float): starting value for the asset simulation
            process_law (str): one of: gaussian, inv_gauss, complex

        Returns:
            pd.DataFrame: asset_df as the final price simulation for every asset
                          X_df as the underlying base process generating the simulation
        """
        X_df = pd.DataFrame()
        X_tilde_df = pd.DataFrame()
        asset_df = pd.DataFrame()
        for asset_nr in range(number_of_assets):
            # Generate iid random variables for a stochastic process
            if process_law == "inv_gauss":
                # follow p.66
                dist = scipy.stats.norminvgauss(a=mean_process, b=sd_process)
                try:
                    X = dist.rvs(len_process-1)
                except:
                    print("Ensure scale parameter is positive for all\
                          distributions, might lead to negative values with current sd")
                    break
            elif process_law == "complex":
                # Model Z iid acc. to p.66
                Z = np.random.normal(loc=mean_process, scale=sd_process, size=len_process-1)
                # Model the sigma process acc. to p.66
                sigma = np.random.normal(loc=mean_process+1, scale=sd_process+1, size=len_process-1)
                # Generate new X process combining both acc. to p.66
                X = Z * sigma
            else:
                X = np.random.normal(loc=mean_process, scale=sd_process, size=len_process-1)
            # Set X_0 = 0 acc. to p.62
            X = np.append(0, X)
            X_df[f"X_process_{asset_nr}"] = X
            # acc. to p.62
            X_tilde = np.cumsum(np.exp(X)-1)
            X_tilde_df[f"X_tilde_process_{asset_nr}"] = X_tilde
            # Plug tuned X_deltas into stochastic exponential formula acc. to p.62
            asset_df[f"asset_process_{asset_nr}"] = (X_tilde_df[f"X_tilde_process_{asset_nr}"].diff().apply(lambda x: 1+x).cumprod() * S_0_1) \
                                                    .dropna()
        return asset_df, X_df

    asset_df, X_df = generate_stochastic_exponential_process(number_of_assets=number_of_assets,
                                                       len_process=len_process,
                                                       sd_process=sd_process,
                                                       mean_process=mean_process,
                                                       S_0_1=S_0_1,
                                                       process_law=process_law)

    def plot_stochastic_exponential_process(asset_df: pd.DataFrame, X_df: pd.DataFrame):
        fig = plt.figure(constrained_layout=True, figsize=(13, 6))
        subplots = [["Top", "Top"],
                    ["Bottom", "Bottom"]]
        axs = fig.subplot_mosaic(subplots)
        for col_name in asset_df.columns:
            axs['Top'].plot(asset_df[col_name], label=col_name)
        axs['Top'].set_title(f"Asset Price Simulation with SD: {sd_process} and Mean: {mean_process} for {asset_df.shape[0]+1} Trading Days for {asset_df.shape[1]} assets",
                # fontdict={"size": 8}
                )
        axs['Top'].set_ylabel('Pice')
        axs['Top'].set_xlabel(f'{asset_df.shape[0]+1} Trading Days')
        axs['Top'].grid(zorder=0)
        # Plot the distribution of simulated daily returns
        for col_name in X_df.columns:
            axs['Bottom'].scatter(y=X_df[col_name].values, x=X_df.index, label=col_name, marker=".")
        axs['Bottom'].set_title('Simulation of daily normally distributed returns')
        axs['Bottom'].set_xlabel(f'{asset_df.shape[0]+1} Trading Days')
        axs['Bottom'].grid(zorder=0)
        plt.show()


    def plot_hist(X_df: pd.DataFrame, actual_returns: pd.DataFrame):
        fig = plt.figure(constrained_layout=True, figsize=(10, 6))
        subplots = [["Top", "Top"],
                    ["Bottom", "Bottom"]]
        axs = fig.subplot_mosaic(subplots)
        axs["Top"].hist(X_df, stacked=True, bins=50, density=True)
        axs['Top'].set_title(f"Distribution for {X_df.shape[1]} Simulations")
        axs['Top'].grid(zorder=0)
        # Now plot the actual realized distribution
        axs["Bottom"].hist(actual_returns, stacked=True, bins=50, density=True)
        axs['Bottom'].set_title("Distribution for realized distribution of returns")
        axs['Bottom'].grid(zorder=0)
        plt.show()

    plot_stochastic_exponential_process(asset_df=asset_df, X_df=X_df)

    # Finally use the simulation technique for actual DAX Index Data and create
    # simulations based on the realized parameters
    # Fit the actual DAX parameters
    dax_data = yf.download(DAX_yf_ticker)
    # Compute the returns
    dax_returns = dax_data["Adj Close"].pct_change().dropna()
    # Get the reaöized parameters
    dax_realized_sd = dax_returns.describe()["std"]
    dax_realized_mean = dax_returns.describe()["mean"]
    dax_realized_kurt = scipy.stats.kurtosis(dax_returns)
    dax_realized_skew = scipy.stats.skew(dax_returns)
    dax_starting_value = dax_data.iloc[0,:]["Adj Close"]
    number_of_assets = 10
    len_process = dax_data.shape[0]

    asset_df, X_df = generate_stochastic_exponential_process(number_of_assets=number_of_assets,
                                                        len_process=len_process,
                                                        sd_process=dax_realized_sd,
                                                        mean_process=dax_realized_mean,
                                                        S_0_1=dax_starting_value,
                                                        process_law=process_law)
    plot_stochastic_exponential_process(asset_df=asset_df, X_df=X_df)
    # also compare the realized distribution with the simuated ones
    plot_hist(X_df=X_df, actual_returns=dax_returns)

if __name__ == "__main__":
    main()
