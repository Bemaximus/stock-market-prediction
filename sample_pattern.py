import numpy as np

# create a method for each candlestick pattern
# that returns the probability of that pattern
# existing right now in a given stock

def stock_went_down_today(OPEN, HIGH, LOW, CLOSE, VOL):
	"""
	Check if a given stock decreased in price since the previous day
	Today's prices can be indexed as [-1], yesterday as [-2], etc.
	"""
	prev_close = CLOSE[-2]
	today_close = CLOSE[-1]

	if today_close < prev_close:
		return 1
	else:
		return 0