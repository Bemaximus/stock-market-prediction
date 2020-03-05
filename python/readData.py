import numpy as np
import pandas as pd
import sys

ticker = "JNUG"

def loadData():
	"""
	With a given ticker, load the data as a pandas dataframe and return it

	"""
	global ticker
	stockData = pd.read_csv(f"../data/{ticker}.csv", index_col="Date")
	return stockData


def saveData(Y, C, T, Q):
	modelDict = {
		"Y": Y,
		"C": C,
		"T": Q,
		"Q": Q
	}

	pass

if __name__ == "__main__":

	# Assign the ticker
	global ticker
	if len(sys.argv) >= 2:
		ticker = sys.argv[1]
	else:
		ticker = input("What stock do you want to test?")

	stockData = loadData()

