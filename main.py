import sys
import datetime
import random

from predict_market import backtest, est_perc_increase
from read_data import load_data, save_data


if __name__ == "__main__":
	"""
	Run this model on a given stock with a given opening price
	"""

	# Assign stock ticker variable and opening price (if necessary)
	if len(sys.argv) >= 2:
		ticker = sys.argv[1]
	else:
		ticker = input("What stock do you want to test? ")

	if len(sys.argv) >= 3:
		open_price = sys.argv[2]
	else:
		open_price = input("What is the opening price of that stock?\n" + 
			"Or type 'None' for no opening price ")
		open_price = None if open_price in ("None", "") else float(open_price)
	# ticker and open_price are assigned now

	stock_data = load_data(ticker)
	save_data(stock_data)
	backtest(stock_data, ticker)
	est_perc_increase(ticker, open_price)

