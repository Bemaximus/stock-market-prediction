import numpy as np
import pandas as pd
import sys

def load_data(ticker):
	"""
	With a given ticker, load the data as a pandas dataframe and return it

	"""
	stock_data = pd.read_csv(f"../data/{ticker}.csv", index_col="Date")
	return stock_data


def save_data(Y, C, T, Q):
	"""
	Save the model testing and training data to a file
	Pickle? JSON? Not sure at the moment

	"""

	model_dict = {
		"Y": Y,
		"C": C,
		"T": Q,
		"Q": Q
	}

	pass

if __name__ == "__main__":

	# Assign the ticker
	if len(sys.argv) >= 2:
		ticker = sys.argv[1]
	else:
		ticker = input("What stock do you want to test? ")

	stock_data = load_data(ticker)

