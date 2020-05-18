# Sample backtesting script using Quantiacs
import numpy as np

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
	"""
	Given the OHLC data of the day, run the algorithm
	Return a list of weights for stocks to invest in
	ex. [0.2, 0.5, -0.3] if you want 
	20% stock 1, 50% in stock 2, and 30% shorted in stock 3
	
	"""

	# CLOSE (and all OHLC) are matrices
	# Each column is a different stock
	# Each row is a different day
	nMarkets = CLOSE.shape[1]

	# Assign weights for different stocks in the portfolio
	# In the same order as the columns in nMarkets
	weights = np.zeros(nMarkets)
	weights[1] = 1
	
	return weights, settings



def mySettings():
	"""
	Create a list of settings for use during backtesting
	Returns a dictionary of parameters
	"""

	settings = dict()

	# The 'markets' attribute is a list of stocks to trade from
	# Quantiacs only supports some equities and futures,
	# but it should be enough to test our day trading algorithms
	# CASH is always the first argument, and this is just how much
	# money we keep in cash
	settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
	'AMGN','AMZN','COST','CSCO','CVS','GE','GILD','GM','GOOGL','MSFT','T']

	# This is the amount of days of data we get in the algorithm. For example, 
	# if we use the last 10 days of data to make trading decisions
	# the lookback is 10
	settings['lookback'] = 10

	# starting money
	settings['budget'] = 1000

	# uncertainty (idk the exact financial definition)
	settings['slippage'] = 0.05

	return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
	import quantiacsToolbox
	results = quantiacsToolbox.runts(__file__)
	# possibly pickle this?