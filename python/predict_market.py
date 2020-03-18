import numpy as np
import pandas as pd
import random
from datetime import datetime
import scipy.linalg as la

def backtest(Y, C, T, Q, ticker):
	"""
	Using training and testing data,
	develop a model to predict the
	stock market and test that model

	Params:
		Y: Matrix of train data
		C: "Solutions" to the train data
		T: Matrix of test data
		Q: "Solutions" to the test data

	Returns:
		P: Portfolio gain over test data 
			(but as a multiplying
			factor, not a percentage)
		A: Portfolio gain every day for 
			the test data

	"""

	pass

def train_model(ticker):
	"""
	Train a model 

	"""

	pass

def est_perc_increase(ticker, opening_price, date=datetime.today()):
	"""
	Given a stock and its opening price, predict the percentage
	increase from opening price to closing price.
	A value of 1 means the stock means no change.

	Params:
		ticker: the stock ticker
		opening_price: the opening price / current price
		date: the day of the transaction
	
	Returns:
		the percentage increase of the day
		currently returns a random number
	"""


	return random.uniform(0.95, 1.05)

if __name__ == "__main__":
	pass