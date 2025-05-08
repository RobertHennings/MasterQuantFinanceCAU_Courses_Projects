import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

"""
Description:
In what follows is the implementation of the simplest way to generate martingale
processes, based on the model on page 36, what is just a simple summation of
random samples drawn from a gaussian law, what has the needed property E(X_n)=0,
so specify loc=0.
"""

def main():
    len_martingale = 1000
    number_martingales = 60
    sd_processes = 3 # 1 for standard normal
    return_stochastic_process_df = True
    show_mean = True

    def generate_martingales(len_martingale: int, number_martingales: int,
                             sd_processes: float, return_stochastic_process_df: bool) -> pd.DataFrame:
        """Generates the simplest form of martingales using i.i.d random variables
           drawn from the standard normal distribution (centered at 0) and takes
           the cumulative sum of them at every discrete time point n

        Args:
            len_martingale (int): length of discrete time steps n
            number_martingales (int): number of martingales that should be created
            return_stochastic_process_df (bool): the random numbers the martingales
                                                 are based on

        Returns:
            pd.DataFrame: holds the martinagles as columns, optionally returns the
                          raw stochastic process matrix as well
        """
        martingale_df = pd.DataFrame()
        stochastic_process_df = pd.DataFrame()

        for martingale_nr in range(number_martingales):
            # acc. to p.36 draw random samples with property E(X_n)=0
            stochastic_process = np.random.normal(
                loc=0, scale=sd_processes, size=len_martingale).tolist()
            stochastic_process_df[f"stochastic_process_{martingale_nr}"] = stochastic_process
            # acc. to p.36 Example 3.10)
            martingale_df[f"martingale_{martingale_nr}"] = stochastic_process_df[f"stochastic_process_{martingale_nr}"].cumsum(
            )
        martingale_df["martingale_mean"] = martingale_df.mean(axis=1)
        return martingale_df if return_stochastic_process_df == False else martingale_df, stochastic_process_df

    martingale_df, stochastic_process_df = generate_martingales(len_martingale=len_martingale,
                                                                number_martingales=number_martingales,
                                                                sd_processes=sd_processes,
                                                                return_stochastic_process_df=return_stochastic_process_df)

    def plot_martingales(martingale_df: pd.DataFrame, show_mean: bool):
        """creates a subplot showing the martingales themselves with mean at every
           discrete time point n and the mean as standalone plot as option

        Args:
            martingale_df (pd.DataFrame): martingale matrix holding the single
                                          martingales as columns
            show_mean (bool): show mean in its own plot or not
        """
        means_N = martingale_df.tail(1).T
        means_N.rename({martingale_df.shape[0]-1: f"Means in N={len_martingale}"}, axis=1, inplace=True)

        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        if show_mean is True:
            subplots = [['TopLeft', 'TopRight'],
                        ['Bottom', 'Bottom']]
        else:
            subplots = [['TopLeft', 'TopRight']]
            
        axs = fig.subplot_mosaic(subplots,
                                gridspec_kw={'width_ratios':[2, 1]})

        # the martingale plot
        axs['TopLeft'].set_title(f"{len(martingale_df.columns)-1} Martingales and Mean (red)")
        for col_name in martingale_df.columns:
            axs['TopLeft'].plot(martingale_df[col_name], label=col_name)
        axs['TopLeft'].plot(martingale_df["martingale_mean"], label="martingale_mean",
                           linestyle="-", color="red", linewidth=3)
        axs['TopLeft'].grid(zorder=0)
        axs['TopLeft'].set_xlabel('Discrete Periods')
        # the distribution of the terminal values of the mean
        axs['TopRight'].set_title(f'Distribution of means at terminal N={martingale_df.shape[0]}')
        axs['TopRight'].hist(means_N, density=True, orientation='horizontal', range=(axs['TopLeft'].get_ylim()[0], axs['TopLeft'].get_ylim()[1]))
        axs['TopRight'].grid(zorder=0)
        # axs['TopRight'].xticks(rotation=45)
        ax_top_right_kde = axs['TopRight'].twinx()
        sns.kdeplot(means_N.iloc[:,0], vertical=True, ax=ax_top_right_kde, color="orange").set(ylabel=None, yticks=[],
                                                                                            #    xticklabels={"rotation":45},
                                                                                               ylim=(axs['TopLeft'].get_ylim()[0],
                                                                                                     axs['TopLeft'].get_ylim()[1])
                                                                                               )
        # ax_top_right_kde.tick_params(axis='x', rotation=45)
        axs['TopRight'].set_xlabel('KDE and Density')
        # the mean standalone
        if show_mean is True:
            axs['Bottom'].set_title('Mean of martingale values')
            axs['Bottom'].set_xlabel('Discrete Periods')
            axs['Bottom'].plot(martingale_df["martingale_mean"], label="martingale_mean",
                                linestyle="-", color="red", linewidth=3)
            axs['Bottom'].grid(zorder=0)
        plt.show()

    plot_martingales(martingale_df=martingale_df, show_mean=show_mean)


if __name__ == "__main__":
    main()
