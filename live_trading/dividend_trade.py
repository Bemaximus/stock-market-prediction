import sys

sys.path.insert(0, './config')
sys.path.insert(0, '../strategies/dividends')

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
from alt_dividend_sort import highest_paying_tickers

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

def opening_buys(account_money=None):
	"""
	Using the opening price and 2 weeks of historical data, choose what to buy

	"""
	# Amount of money to invest
	if account_money == None:
		account_money = float(api.get_account().cash)

	tickers_to_buy = tuple(np.unique(highest_paying_tickers()["Ticker"].to_numpy()))
	num_tickers = tickers_to_buy.shape[0]
	perc_equity = 1 / num_tickers
	orders = [None] * num_tickers

	for i, ticker in enumerate(tickers_to_buy):
		orders[i] = buy_stock_given_equity(ticker, equity=perc_equity, total=account_money)

	bought_bulk_stock_mail(
		tickers=tickers_to_buy,
		orders=orders
	)
	return orders

def buy_stock_given_equity(ticker, equity=1, total=float(api.get_account().cash),
		buffer_perc=0.05, **kwargs):
	"""
	allocate a percentage of the free portfolio 
	"""
	email_content = kwargs["email_content"] if "email_content" in kwargs.keys() else ""
	global api

	price = float(api.get_last_quote(ticker))
	
	# buy this stock
	order = api.submit_order(ticker, (account_money * equity * (1-buffer_perc) // price), 
		"buy", "market", "gtc")

	return order

def liquidate():
	cancel = api.cancel_all_orders() # clear all incomplete orders
	close = api.close_all_positions() # liquidate holdings
	if close:
		liquidate_stock_mail(close) # email notification if there are any trades

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "auto":
	while True:
		d = datetime.datetime.now(pytz.timezone('US/Eastern'))
		try:
			print(d)
			if d.hour == 9 and d.minute == 30:
				market_clock = api.get_clock()
				if market_clock.is_open:
					print('buying now')
					orders = opening_buys()
					print(f"info: {orders}")
				else:
					print('market is closed today')
					send_mail(
						subject="Market closed today",
						body="We are not performing any transactions"
					)
					time.sleep(60 * 60 * 24 - 60 * 2)
					# sleep one day minus two minutes
			elif d.hour == 15 and d.minute == 58:
				print('liquidate')
				liquidate()
		except Exception as e:
			print(e)
			continue
		finally:
			time.sleep(60)
