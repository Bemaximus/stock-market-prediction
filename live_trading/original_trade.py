import sys

sys.path.insert(0, './config')
sys.path.insert(0, '../strategies/QEA')

import datetime
import time
import pytz
import requests
import random
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi

from config import *
from send_mail import *
from predict_market import est_perc_increase

# load the alpaca account
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, api_version='v2', 
	base_url=APCA_API_BASE_URL)

# get account details
account = api.get_account()
print(account)
account_money = float(account.cash)
print(f"${account_money}")

positions = api.list_positions()
print(positions)

# api.submit_order('AAPL',10,'buy','limit','gtc',170.50)

def opening_buys(symbols=["JNUG", "JDST"], account_money=None):
	"""
	Using the opening price and 2 weeks of historical data, choose what to buy

	"""
	# Amount of money to invest
	if account_money == None:
		account_money = float(api.get_account().cash)

	# Store current stock data for sending to the algorithm
	# and preparing transactions
	est_increases = dict()
	current_prices = dict()
	historical_prices = dict()

	# Iterate through each symbol and get
	# a) current price
	# b) last 10 days' quotes
	# c) estimated percent increase
	for symbol in symbols:
		est_increases[symbol], current_prices[symbol], _ = jonas_bonus(symbol)
		# current_prices[symbol] = float(api.alpha_vantage.current_quote(symbol)["05. price"])
		print(f"{symbol}: ${current_prices[symbol]}")
		# est_increases[symbol] = random.uniform(0.95, 1.05) # est_perc_increase(symbol, current_prices[symbol])

	buy_ticker = max(est_increases, key=est_increases.get)
	print(buy_ticker)
	print(est_increases[buy_ticker])
	if est_increases[buy_ticker] > 1:
		# buy this stock
		r = api.submit_order(buy_ticker, (account_money * .95) // current_prices[buy_ticker], 
			"buy", "market", "gtc")
		print(account_money // current_prices[buy_ticker])
		bought_stock_mail(r.symbol, r.qty, price=current_prices[buy_ticker],
			trade=f"Buy: {buy_ticker}\nEstimated Increases: {est_increases}\n" + 
			f"Current Prices: {current_prices}\n{r}")
		return r
	return 0

def liquidate():
	cancel = api.cancel_all_orders() # clear all incomplete orders
	close = api.close_all_positions() # liquidate holdings
	if close:
		liquidate_stock_mail(close) # email notification if there are any trades

def jonas_bonus(ticker):
	"""
	Take a ticker and return the current price and percentage increase
	Make an API call to Alphavantage to receive this data
	"""
	params = {
		"symbol": ticker,
		"adjusted": False,
		"outputsize": "compact",
		"cadence": "daily",
		"output_format": "pandas"
	}
	historical_dataset = api.alpha_vantage.historic_quotes(**params)

	last_ten_days = historical_dataset.iloc[10:0:-1, 0:4].values.flatten()
	opening_price = historical_dataset.iloc[0, 0]
	current_price = historical_dataset.iloc[0, 3]
	normalizing_factor = last_ten_days[0]

	algo_inputs = np.append(last_ten_days, opening_price) / normalizing_factor
	print(algo_inputs)

	price_change = est_perc_increase(ticker, algo_inputs)
	return price_change, current_price, historical_dataset

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "auto":
	while True:
		d = datetime.datetime.now(pytz.timezone('US/Eastern'))
		try:
			print(d)
			if d.hour == 9 and d.minute == 30:
				market_clock = api.get_clock()
				if market_clock.is_open:
					print('buying now')
					stuff = opening_buys(["JNUG", "NUGT", "JDST", "DUST"])
					print(f"info: {stuff}")
				else:
					print('market is closed today')
					send_mail(
						subject="Market closed today",
						body="We are not performing any transactions"
					)
					time.sleep(60 * 60 * 24 - 60 * 5)
					# sleep one day minus five minutes
			elif d.hour == 15 and d.minute == 58:
				print('liquidate')
				liquidate()
		except Exception as e:
			print(e)
			continue
		finally:
			time.sleep(60)
