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

def save_data(stock_data, ticker):
	"""
	Save the model testing and training data as numpy arrays to a Pickle file

	"""
	filtered_stock_data = stock_data.loc[:, ["Open", "High", "Low", "Close"]]
	np_stock_data = filtered_stock_data.to_numpy()
	np_stock_data = np_stock_data[:-250, :]
	# Separate the testing and training data
	print(np_stock_data)
	num_rows = np.size(np_stock_data, 0)
	num_rows_test = 250
	num_rows_train = num_rows - num_rows_test

	Y_unformat = np_stock_data[0:num_rows_train, :]
	T_unformat = np_stock_data[num_rows_train:num_rows, :]

	# Determine the "answers" -> actual percentage increases
	Q = np.divide(T_unformat[10:num_rows,[3]], T_unformat[10:num_rows,[0]])
	C = np.divide(Y_unformat[10:num_rows_train,[3]], Y_unformat[10:num_rows_train,[0]])

	# Initialize the test data arrays
	Y = np.empty((0,41))
	T = np.empty((0,41))

	# Rows to check for invalid data (e.g. splits mess data up)
	Y_valid = np.empty(num_rows_train)
	T_valid = np.empty(num_rows_test)

	# Add training data one row at a time
	for row in range(np.size(Y_unformat, 0)-10):
		first_open = Y_unformat[row,[0]]
		last_open = Y_unformat[row+10,[0]]
		# row 'row+10' isn't used in the line below, only indices 0-9

		unformatted_row = Y_unformat[row:row+10, :]
		formatted_row, Y_valid[row] = normalize_dataset(unformatted_row.copy(), last_open)
		Y = np.append(Y, [formatted_row], axis=0)

	for row in range(np.size(T_unformat, 0)-10):
		first_open = T_unformat[row,[0]]
		last_open = T_unformat[row+10,[0]]

		unformatted_row = T_unformat[row:row+10, :]
		formatted_row, T_valid[row] = normalize_dataset(unformatted_row.copy(), last_open)
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
	#print("input data: ", lookback)
	
	# high, low, close: divide by opening price of the day

	lookback[:, 1] /= lookback[:,0]
	lookback[:, 2] /= lookback[:,0]
	lookback[:, 3] /= lookback[:,0]
	#print("Divide by opening price: ", lookback)

	# save normalized opening price
	opening_price /= lookback[-1, 0]

	# open: divide by the previous open

	lookback[1:, 0] /= lookback[:-1, 0]
	#print("Divide by yesterday's price: ", lookback)

	# first open should be 1
	lookback[0, 0] = 1

	# check if the data are legit
	is_valid = True
	data_variety = np.amax(lookback) / np.amin(lookback)
	if data_variety > 1e4:
		is_valid = False
	#print("data_variety: ", data_variety, "\tis_valid: ", is_valid)

	# flatten the data
	flat_lookback = lookback.flatten()
	#print("Flat lookback shape: ", flat_lookback.shape)
	full_flat_lookback = np.append(flat_lookback, opening_price)

	# subtract 1 from all elements to make the variation more apparent
	full_flat_lookback -= 1

	#print("Processed lookback: ", full_flat_lookback)
	return full_flat_lookback, is_valid

if __name__ == "__main__":

	# Assign the ticker
	if len(sys.argv) >= 2:
		tickers = sys.argv[1:]
	else:
		tickers = [input("What stock do you want to test? ")]
	for ticker in tickers:
		model_dict = save_data(load_data(ticker), ticker)
		print([x.shape for x in model_dict.values()])
