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
	return stock_data

def save_data(stock_data):
	"""
	Save the model testing and training data as numpy arrays to a Pickle file

	"""
	filtered_stock_data = stock_data.loc[:, ["Open", "High", "Low", "Close"]]
	np_stock_data = filtered_stock_data.to_numpy()
	
	# Separate the testing and training data
	print(np_stock_data)
	num_rows = np.size(np_stock_data, 0)
	num_rows_train = num_rows - 250
	Y_unformat = np_stock_data[0:num_rows_train, :]
	T_unformat = np_stock_data[num_rows_train:num_rows, :]

	# Determine the "answers" -> actual percentage increases
	Q = np.divide(T_unformat[10:num_rows,[3]], T_unformat[10:num_rows,[0]])
	C = np.divide(Y_unformat[10:num_rows_train,[3]], Y_unformat[10:num_rows_train,[0]])

	# Initialize the test data arrays
	Y = np.empty((0,41))
	T = np.empty((0,41))

	# Add training data one row at a time
	for row in range(np.size(Y_unformat, 0)-10):
		first_open = Y_unformat[row,[0]]
		last_open = Y_unformat[row+10,[0]]
		# row 'row+10' isn't used in the line below, only indices 0-9
		unformatted_row = Y_unformat[row:row+10, :]
		formatted_row = formatted_row.flatten()
		formatted_row = normalize_dataset(unformatted_row, last_open)
		Y = np.append(Y, [formatted_row], axis=0)
	for row in range(np.size(T_unformat, 0)-10):
		first_open = T_unformat[row,[0]]
		last_open = T_unformat[row+10,[0]]
		formatted_row = T_unformat[row:row+10, :]
		formatted_row = formatted_row.flatten()
		formatted_row = np.append(formatted_row, last_open)
		formatted_row = formatted_row/first_open
		T = np.append(T, [formatted_row], axis=0)
	model_dict = {
		"Y": Y,
		"C": C,
		"T": T,
		"Q": Q
	}
	print(Y.shape)
	print(Y)
	with open(dir_path + f"/data/{ticker}_test_train.p", 'wb') as fp:
		pickle.dump(model_dict, fp)
	return model_dict

def normalize_dataset(lookback, opening_price):
	"""
	Take training data and convert it into a line for the linear regression matrix
	"""
	print(lookback)
	
	# high, low, close: divide by opening price of the day

	lookback[:, 1:] /= lookback[:,0]
	print(lookback)

	# open: divide by the previous open

	lookback[1:, 0] /= lookback[:-1, 0]
	print(lookback)

	# first open should be 1
	lookback[0, 0] = 1

	flat_lookback = lookback.flatten()
	full_flat_lookback = np.append(flat_lookback, opening_price)
	
	return full_flat_lookback

if __name__ == "__main__":

	# Assign the ticker
	if len(sys.argv) >= 2:
		ticker = sys.argv[1]
	else:
		ticker = input("What stock do you want to test? ")
	model_dict = save_data(load_data(ticker))
	print([x.shape for x in model_dict.values()])
