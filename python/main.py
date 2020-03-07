import sys

from predictMarket import predictMarket
from readData import readData






if __name__ == "__main__":
	"""
	Run this model on a given stock with a given opening price
	"""

	# Assign stock ticker variable and opening price (if necessary)
	if len(sys.argv) >= 2:
		ticker = sys.argv[1]
	else:
		ticker = input("What stock do you want to test?")

	if len(sys.argv) >= 3:
		openPrice = sys.argv[2]
	else:
		openPrice = input("What is the opening price of that stock?\n" + 
			"Or type 'None' for no opening price")
		openPrice = None if openPrice in ("None", "") else float(openPrice)

	# ticker and openingPrice are assigned now

