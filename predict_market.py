import numpy as np
import pandas as pd
import random
from datetime import datetime
import scipy.linalg as la
import scipy.io
import json

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
	


def train_model(ticker):
	"""
	Train a model 
	
	"""
	with open(ticker+'_test_train.txt') as json_file:
		model_dict = json.load(json_file)
	Y = np.array(model_dict.get("Y"))
	C = np.array(model_dict.get("C"))
	T = np.array(model_dict.get("T"))
	Q = np.array(model_dict.get("Q"))

	Yt = np.transpose(Y)
	YtY = Yt * Y
	YtC = YT *C
	m = YtC/YtY
	
	return m
	

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
	m = train_model(ticker)
	with open(ticker + '_test_train.txt') as json_file:
		model_dict = json.load(json_file)
		T = np.array(model_dict.get("T"))
	np.insert(T, 0, opening_price)
	G = T[0, 10] * m #Check T index
	return G

	# hack fix to get MATLAB data for stock predictions
	# change this when Python analyses are working
	try:
		ticker_model = scipy.io.loadmat(f"../models/{ticker}_predict.mat")
		m = np.array(ticker_model["m"]).flatten()
		
		historical_data = pd.read_csv(f"../data/{ticker}.csv")
		last_2_weeks_data = historical_data.tail(10)
		last_2_weeks_data = last_2_weeks_data.loc[:,["Open", "High", "Low", "Close"]]

		previous_intraday_data = last_2_weeks_data.values.flatten()
		all_previous_data = np.append(previous_intraday_data, opening_price / previous_intraday_data[0])

		perc_increase = np.prod(m * all_previous_data)
		return perc_increase

	except Exception as e:
		print(e)

	return random.uniform(0.95, 1.05)

if __name__ == "__main__":
	pass