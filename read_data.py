import numpy as np
import pandas as pd
import sys
import os 
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__)) 

def load_data(ticker):
	"""
	With a given ticker, load the data as a pandas dataframe and return it

	"""
	stock_data = pd.read_csv(dir_path + f"/../../data/{ticker}.csv", index_col="Date")
	print(stock_data)
	return stock_data

def save_data(stock_data):
	"""
	Save the model testing and training data as numpy arrays to a Pickle file

	"""
	np_stock_data = stock_data.to_numpy()
	num_rows = np.size(np_stock_data, 0)
	num_rows_train = num_rows - 250
	Y_unformat = np_stock_data[0:num_rows_train, 0:4]
	T_unformat = np_stock_data[num_rows_train-1:num_rows,0:4]
	Q = np.divide(T_unformat[:num_rows-11,[3]], T_unformat[:num_rows-11,[0]])
	C = np.divide(Y_unformat[:num_rows_train-11,[3]], Y_unformat[:num_rows_train-11,[0]])
	Y = np.empty((0,41))
	T = np.empty((0,41))
	for row in range(np.size(Y_unformat, 0)-11):
		first_open = Y_unformat[row,[0]]
		last_open = Y_unformat[row+11,[0]]
		formatted_row = Y_unformat[row:row+10, 0:4]
		formatted_row = formatted_row.flatten()
		formatted_row = np.append(formatted_row, last_open)
		formatted_row = formatted_row/first_open
		Y = np.append(Y, [formatted_row], axis=0)
	for row in range(np.size(T_unformat, 0)-11):
		first_open = T_unformat[row,[0]]
		last_open = T_unformat[row+11,[0]]
		formatted_row = T_unformat[row:row+10, 0:4]
		formatted_row = formatted_row.flatten()
		formatted_row = np.append(formatted_row, last_open)
		#formatted_row = formatted_row/first_open
		T = np.append(T, [formatted_row], axis=0)
	model_dict = {
		"Y": Y,
		"C": C,
		"T": T,
		"Q": Q
	}
	with open(dir_path + f"/data/{ticker}_test_train.p", 'wb') as fp:
		pickle.dump(model_dict, fp)
	return None

if __name__ == "__main__":

	# Assign the ticker
	if len(sys.argv) >= 2:
		ticker = sys.argv[1]
	else:
		ticker = input("What stock do you want to test? ")
	save_data(load_data(ticker))

