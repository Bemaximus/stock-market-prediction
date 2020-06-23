import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import mplfinance as mpf
from datetime import datetime

import os 
dir_path = os.path.dirname(os.path.realpath(__file__)) 

from sample_pattern import stock_went_down_today

# write a function that tests stock data on years of backtested data
def run_pattern(pattern_func, price_matrix, lookback=5) -> np.ndarray:
	"""
	Run a function for a candlestick pattern over long periods of time
	Price matrix is a numpy array with open, high, low, close, and volume
		OLDEST FIRST!
	Return list of probabilities for that pattern over time
	TODO: make this backtest faster

	"""
	_, OPEN, HIGH, LOW, CLOSE, VOL = price_matrix.T # unpack columns from transpose

	# create a vectorized function that does all backtests without a for loop
	def pattern_func_wrapper(i):
		return pattern_func(OPEN[i-lookback+1:i+1], HIGH[i-lookback+1:i+1],
			LOW[i-lookback+1:i+1], CLOSE[i-lookback+1:i+1], VOL[i-lookback+1:i+1])
	vec_pattern_func = np.vectorize(pattern_func_wrapper)

	# all days to test for
	lookback_list = range(lookback-1, price_matrix.shape[1])

	# run the function over the days being tested
	pattern_output_list = vec_pattern_func(lookback_list)
	return pattern_output_list

def plot_pattern(ticker, pattern_func=stock_went_down_today, 
		start_date=None, end_date=None, lookback=5):
	
	# sanitize datetime data
	# if isinstance(start_date, str):
	# 	start_date = datetime.strptime(start_date, '%Y-%m-%d')
	# if isinstance(end_date, str):
	# 	end_date = datetime.strptime(end_date, '%Y-%m-%d')
	
	# get stock data
	filtered_stock_data = get_filtered_data(ticker, start_date, end_date)
	price_matrix = filtered_stock_data.to_numpy()

	# run algorithm
	# pattern_output_list = run_pattern(pattern_func, price_matrix, lookback)

	# plot data
	mpf.plot(filtered_stock_data, type="candle", style="charles", volume=True)

	pass


def get_filtered_data(ticker, start_date=None, end_date=None):
	"""
	With a given ticker, load the data as a pandas dataframe and return it

	"""
	stock_data = pd.read_csv(dir_path + f"/../../data/{ticker}.csv",
		 index_col="Date", parse_dates=True)
	filtered_stock_data = stock_data.loc[start_date:end_date,
		["Open", "High", "Low", "Close", "Volume"]]
	filtered_stock_data
	return filtered_stock_data