import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from cli_printing import print_progress_bar

import time
import lxml
import requests
import re

MARKETS = ["NYSE", "NASDAQ"]

def get_all_data(ticker_list):
	ticker_list_length = ticker_list.shape[0]
	ticker_stats = pd.Series([None] * ticker_list_length)
	for i, ticker in ticker_list.iteritems():
		
		print_progress_bar(i, ticker_list_length, fraction=True, 
			prefix="Checking Tickers:", suffix="Tickers checked")
		ticker_stats[i] = get_valid_url(ticker)

	print_progress_bar(ticker_list_length, ticker_list_length, fraction=True, 
		prefix="Checking Tickers:", suffix="Completed", printEnd="\n")
	return ticker_stats


def get_valid_url(ticker):
	"""
	Try to get a valid url for scraping a ticker on MarketBeat
	"""
	# print(ticker)

	possible_ticker_urls = (f"https://www.marketbeat.com/stocks/{m}/{ticker}/dividend/" for m in MARKETS)

	for url in possible_ticker_urls:
		response = requests.get(url)
		
		if response.status_code == 200:
			break;
	else:
		return None

	soup = BeautifulSoup(response.content, "lxml")
	ticker_stats = soup.select("#cphPrimaryContent_tabDividendHistory div.price-data strong")
	# print(ticker_stats)

	if len(ticker_stats) != 7:
		return None

	dividend_occurence = ticker_stats[5].text
	# print(dividend_occurence)

	return dividend_occurence
	

if __name__ == "__main__":
	data_df = pd.read_csv("all_thestreet_dividends.csv")
	data_df["Occurence"] = get_all_data(data_df["Symbol"])