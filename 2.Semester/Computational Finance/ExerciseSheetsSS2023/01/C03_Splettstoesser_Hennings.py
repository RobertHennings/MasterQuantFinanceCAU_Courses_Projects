# Sheet: 01, CF 2023
# Group: QF 27
# Group Members: Tomke Splettstoesser, Robert Hennings

import numpy as np
import matplotlib.pyplot as plt

# a) Define function log_returns(data)
def log_returns(data: np.array) -> np.array:
    """Function that expects a list/array like type of data to compute log returns from
       Important: the data should be in the correct (timely) order so that the returns make sense

    Args:
        data (np.array): raw price column data (close) from that log returns should be computed

    Returns:
        np.array: log returns of the raw price data (close)
    """
    print("Initial data array has length: ", len(data))
    log_return_array = np.diff(np.log(data))
    print("Log return data array has length: ", len(log_return_array))
    return log_return_array

# b)
# Test function with time_series_dax_2023.csv data
# Import data
file_path = "//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//Computational Finance//ExerciseSheets//"
file = "time_series_dax_2023.csv"

dax = np.flip(np.genfromtxt(file_path+file,
                    delimiter=";",
                    usecols=(4),
                    skip_header=1))

# Test function
log_returns(data=dax)
LogR_Dax = log_returns(data=dax)
# Plot log return time series
plt.plot(LogR_Dax, color="#603086")
plt.ylabel("Log returns of DAX Closing Price Time Series")
plt.xlabel("Index of Data point", fontsize=8)
plt.title("Log return of DAX Closing prices", fontsize=14)
plt.xticks(rotation=45, fontsize=6)
plt.show()

# Compute and display the annualized empirical mean and standard deviation of log-returns
def annualized_empirical_mean(data: np.array, trading_days: int) -> float:
    """Function that computes the annualized empirical mean of a
       list/array like type format of data
       Important: Since we already shortened the data in the function log_returns
                  we just sum up all the values directly and not start at the
                  second return

    Args:
        data (np.array): float data points
        trading_days (int): assumed number of trading days per year for annualization

    Returns:
        float: annualized empirical mean
    """
    mu_hat = (trading_days/(len(data)-1)) * sum(data)
    print("Annualized empirical mean of data: ", mu_hat)
    return mu_hat


def annualized_empirical_sd(data: np.array, trading_days: int) -> float:
    """Function that computes the annualized empirical sd of a
       list/array like type format of data
       Important: Since we already shortened the data in the function log_returns
                  we just sum up all the values directly and not start at the
                  second return

    Args:
        data (np.array): float data points
        trading_days (int): assumed number of trading days per year for annualization

    Returns:
        float: annualized empirical standard deviation
    """
    mu_hat = (trading_days/(len(data)-1)) * sum(data)
    sd_hat = np.sqrt((trading_days / (len(data)-2)) * sum((data - (mu_hat / trading_days))**2))
    print("Annualized empirical standard deviation of data: ", sd_hat)
    return sd_hat


# Test functions
annualized_empirical_mean(data=log_returns(data=dax), trading_days=250)
# 0.06387663982975988
# What should be (nearly) equivalent to:
print('Annualized empirical mean = ', 250*(np.mean(LogR_Dax)))
# 0.06386910809531468
annualized_empirical_sd(data=log_returns(data=dax), trading_days=250)
# 0.22200641743326377
# What should be (nearly) equivalent to:
print('Annualized empirical standard deviation = ', np.sqrt(250*(np.var(LogR_Dax, ddof=1))))  # ddof=1 for unbiased estimator
# 0.2219933270457939

# c)
# Simulate a time serie of log returns
# assumptions: normally distributed
# Input parameters: computed empirical mean and sd


def simulate_log_returns(mean: float, sd: float, trading_days: int, seed_state: int, num_sim: int) -> np.array:
    """Function that draws a random sample of size sum_sim from a normal distribution with the passed parameters
       mean, sd and returns daily data points 
       Important: trading_days is used to rescale to a daily frequency

    Args:
        mean (float): mean for the normal distribution from which the random sample is drawn
        sd (float): sd for the normal distribution from which the random sample is drawn
        trading_days (int): assumed number of trading days
        seed_state (int): seed for reproducability
        num_sim (int): number of data points to simulate

    Returns:
        np.array: simulated data points from the specified distribution
    """
    # First calculate the scaling factors on a daily frequency to meet the comparison requirement
    # daily mean
    daily_mean = mean/trading_days
    # daily standard_deviation
    # Adjust: daily_avg * np.sqrt(trading_days) = trading_day_average
    # So wee just apply: daily_avg = trading_day_average / np.sqrt(trading_days)
    # but just holds if we assume that the single returns are i.i.d
    daily_sd = sd / np.sqrt(trading_days)

    # Draw random numbers that are assumed to be normally distributed
    # with the daily_mean as mean and daily_sd as sd
    # for reproducability
    np.random.seed(seed_state)
    simulated_log_returns = np.random.normal(loc=daily_mean,
                                             scale=daily_sd,
                                             size=num_sim)
    return simulated_log_returns


simulate_log_returns(mean=annualized_empirical_mean(data=LogR_Dax, trading_days=250),
                     sd=annualized_empirical_sd(data=LogR_Dax, trading_days=250),
                     trading_days=250,
                     seed_state=1,
                     num_sim=len(LogR_Dax))

sim_log_rets = simulate_log_returns(mean=annualized_empirical_mean(data=LogR_Dax, trading_days=250),
                                    sd=annualized_empirical_sd(data=LogR_Dax, trading_days=250),
                                    trading_days=250,
                                    seed_state=1,
                                    num_sim=len(LogR_Dax))

# Plot simulated log-returns together with atcual log-return dax data
fig, ax = plt.subplots()
plt.plot(LogR_Dax, color="#603086", label="Log DAX")
plt.plot(sim_log_rets, color="black", label="Sim.", alpha=0.5)
plt.axhline(0, color='black')
plt.ylabel("Log returns")
plt.xlabel("Index of Data point", fontsize=6)
plt.title("Log return of DAX Closing prices vs. Simulation", fontsize=14)
plt.xticks(rotation=45, fontsize=6)
plt.legend()
plt.show()

# Interpretation:
"""
As we can see from the comparison of the log returns side by side they differ very
much in that sense, that the simulated data spreads far more wide across the whole
observed sample timeframe. Although it misses to incorporate also bigger spikes
(positive and negative) that occur in the real data displayed. These extreme
values also only occur in specific time periods and not always what
can be summerasied by the principle of volatility clustering.
Additionally we saw in the raw series trends and dependency structures, that
point to non-stationarity. This characteristic can be corrected
(like here aplying a log transformation) but it is still evident that
the effect isnt completely gone.
"""

# d) Differences between actual log return data and simulated log return data
fig, ax = plt.subplots()
plt.hist(LogR_Dax, color="#603086", label="Log DAX", density=True, bins=100)
plt.hist(sim_log_rets, color="black", label="Sim.", density=True, bins=100, alpha=0.5)
plt.axvline(annualized_empirical_mean(data=log_returns(data=dax), trading_days=250)/250, color='r', label='Emp. mean')
plt.ylabel("Density")
plt.xlabel("Values", fontsize=8)
plt.title("Density of Log return of DAX Closing prices vs. Simulation", fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Interpretation:
"""
When comparing the histogram of the real world data and the simulated data we can see
the fat tails characteristic of the real world data. If we zoom in at the very
ends of both sides of the probability distribution, we can see that the reald 
world data has way more probability mass in these areas, than the simulation
would assume.    
"""