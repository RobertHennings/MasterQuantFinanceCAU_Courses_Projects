import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
import plotly.express as px
"""
Description:
In what follows is the implementation of the brownian motion and the brownian
motion with drift according to the equations on page 125-127. A Brownian Motion
is itself a subclass of processes of the Levy Process family, where the sd is ever
increasing over time with a constant mean of 0.
Definition 8.2. A Lévy process is called standard Brownian motion if X1 is a
standard normal random variable, i.e. with mean 0 and variance 1.
The law of standard Brownian motion is uniquely determined. We have that Xt is
normally distributed with mean 0 and variance t.
"""

def main():
    number_processes = 20
    mu = 0.8
    sd = 1.2
    N = 1
    # needed for discretization
    steps_process = 1000


    def generate_brownian_motion(number_processes: int, N: int, steps_process: int) -> pd.DataFrame:
        bm_df = pd.DataFrame()
        dt = (N/steps_process)
        for process_nr in range(number_processes):
            # Acc. to p.126 Definition 8.2) and Remark
            bm = np.cumsum([np.random.normal(loc=0, scale=np.sqrt(t), size=1) for t in np.arange(0, N, dt)])
            bm_df[f"bm_process_{process_nr}"] = bm

        bm_df["bm_mean"] = bm_df.mean(axis=1)
        return bm_df


    bm_df = generate_brownian_motion(number_processes=number_processes,
                                    N=N, steps_process=steps_process)


    def generate_brownian_motion_with_drift(number_processes: int,
                                            mu: float, sd: float, N: int, steps_process: int) -> pd.DataFrame:
        bmwd_df = pd.DataFrame()
        dt = (N/steps_process)
        for process_nr in range(number_processes):
            # acc. to p.126, Theorem 8.3) and Defintion 8.4)
            bmwd = np.cumsum([(mu*0.1)+sd*np.random.normal(loc=0, scale=np.sqrt(t), size=1) for t in np.arange(0, N, dt)])
            bmwd_df[f"bmwd_process_{process_nr}"] = bmwd
            
        bmwd_df["bmwd_mean"] = bmwd_df.mean(axis=1)
        return bmwd_df


    bmwd_df = generate_brownian_motion_with_drift(number_processes=number_processes,
                                        mu=mu, sd=sd, N=N, steps_process=steps_process)


    def plot_bm_bmwd_process(bm_df: pd.DataFrame, bmwd_df: pd.DataFrame):
        fig = plt.figure(constrained_layout=True, figsize=(13, 6))
        subplots = [["Top", "Top"],
                    ["Bottom", "Bottom"]]
        axs = fig.subplot_mosaic(subplots)
        # Brownian Motion
        for col_name in bm_df.columns:
            axs['Top'].plot(bm_df[col_name], label=col_name)
        # color mean in red
        axs['Top'].plot(bm_df["bm_mean"], label="bm_mean",
                        color="red", linewidth=3)
        axs['Top'].set_title(f"Simulation of a Brownian Motion process for a discretization of {bm_df.shape[0]} steps for N {N} periods for {bm_df.shape[1]} assets, mean in red",
                )
        axs['Top'].set_ylabel('Pice')
        axs['Top'].set_xlabel(f'{bm_df.shape[0]} steps for discretized Time N {N}')
        axs['Top'].grid(zorder=0)
        # Brownian Motion with drift
        for col_name in bmwd_df.columns:
            axs['Bottom'].plot(bmwd_df[col_name], label=col_name)
        # color mean in red
        axs['Bottom'].plot(bmwd_df["bmwd_mean"], label="bmwd_mean",
                        color="red", linewidth=3)
        axs['Bottom'].set_title(f'Simulation of Brownian Motion with drift with parameters mu: {mu} and sd: {sd} for {bmwd_df.shape[1]} assets, mean in red')
        axs['Bottom'].set_ylabel('Pice')
        axs['Bottom'].set_xlabel(f'{bmwd_df.shape[0]} steps for discretized Time N {N}')
        axs['Bottom'].grid(zorder=0)
        plt.show()


    plot_bm_bmwd_process(bm_df=bm_df, bmwd_df=bmwd_df)

    # Since the single inceremts are independent of each other and only show dependence
    # with the previous single Lag, the pacf should show only +- spikes for the first
    # preceeding Lag what it indeed does looking at the resulting PACF, the ACF shows
    # a high dependence throughout multiple Lags
    acf_pacf_fig(bmwd_df.bmwd_process_0, both=True, lag=20)
    plt.show()

    # Same holds for the Brownian Motion process itself
    acf_pacf_fig(bm_df.bm_process_0, both=True, lag=20)
    plt.show()

    def plot_2d_bm_bmwd_process(bm_df: pd.DataFrame, bmwd_df: pd.DataFrame, N: int, mu: float, sd: float):
        if bm_df.shape[1] >= 2 and bmwd_df.shape[1] >= 2:
            fig = plt.figure(constrained_layout=True, figsize=(12, 12))
            subplots = [["Top", "Top"],
                        ["Bottom", "Bottom"]]
            axs = fig.subplot_mosaic(subplots)
            # Brownian Motion
            axs['Top'].plot(bm_df[bm_df.columns[0]], bm_df[bm_df.columns[1]])
            axs['Top'].set_title(f"Simulation of a 2D Brownian Motion process for a discretization of {bm_df.shape[0]} steps for N {N} periods",
                    )
            axs['Top'].set_ylabel(f'{bm_df.columns[1]}')
            axs['Top'].set_xlabel(f'{bm_df.columns[0]}')
            axs['Top'].grid(zorder=0)
            # Brownian Motion with drift
            axs['Bottom'].plot(bmwd_df[bmwd_df.columns[0]], bmwd_df[bmwd_df.columns[1]])
            axs['Bottom'].set_title(f'Simulation of a 2D Brownian Motion with drift with parameters mu: {mu} and sd: {sd}')
            axs['Bottom'].set_ylabel(f'{bmwd_df.columns[1]}')
            axs['Bottom'].set_xlabel(f'{bmwd_df.columns[0]}')
            axs['Bottom'].grid(zorder=0)
            plt.show()
        else:
            print("Not enough processes to plot in a 2d plot, provide at least 2 for each df")


    plot_2d_bm_bmwd_process(bm_df=bm_df,
                            bmwd_df=bmwd_df,
                            N=N, mu=mu, sd=sd)


    def plot_3d_process(process_df: pd.DataFrame, title: str):
        if process_df.shape[1] >=3:
            fig_3d_process = px.line_3d(process_df, x=process_df.columns[0], y=process_df.columns[1], z=process_df.columns[2])
            fig_3d_process.update_layout({"title": title})
            fig_3d_process.show()
        else:
            print("Not enough processes to plot in a 3d plot, provide at least 3 processes")


    plot_3d_process(process_df=bmwd_df,
                    title=f"Simulation of a 3D Brownian Motion with drift with parameters mu: {mu} and sd: {sd}")


    plot_3d_process(process_df=bm_df,
                    title=f"Simulation of a 3D Brownian Motion")


if __name__ == "__main__":
    main()
