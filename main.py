import sys
import datetime
import random

from predict_market import backtest, est_perc_increase, train_model
from read_data import load_data, save_data


if __name__ == "__main__":
	"""
	Run this model on a given stock with a given opening price
	"""

	# Assign stock ticker variable and opening price (if necessary)
	if len(sys.argv) >= 2:
		tickers = sys.argv[1:]
	else:
		tickers = [input("What stock do you want to test? ")]

	print_this = ""
	for ticker in tickers:
		stock_data = load_data(ticker)
		save_data(stock_data, ticker)
		train_model(ticker)
		print_this += f"{ticker}:\t{backtest(ticker)[1]}\n"
	print(print_this)
