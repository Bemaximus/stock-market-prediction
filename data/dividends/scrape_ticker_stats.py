import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from cli_printing import print_progress_bar

import time
import lxml
import requests
import re

MARKETS = ["NYSE", "NASDAQ"]

def get_all_marketbeat_periods(ticker_list):
	ticker_list_length = ticker_list.shape[0]
	ticker_stats = pd.Series([None] * ticker_list_length)
	for i, ticker in ticker_list.iteritems():
		
		print_progress_bar(i, ticker_list_length, fraction=True, 
			prefix="Checking Tickers:", suffix="Tickers checked")
		ticker_stats[i] = get_valid_url(ticker)

	print_progress_bar(ticker_list_length, ticker_list_length, fraction=True, 
		prefix="Checking Tickers:", suffix="Completed", printEnd="\n")
	return ticker_stats


def get_marketbeat_period(ticker):
	"""
	Try to get a valid url for scraping a ticker on MarketBeat
	"""

	# Go to the url for this stock
	url = f"https://www.marketbeat.com/stocks/NASDAQ/{ticker}/dividend/"

	response = requests.get(url)
	
	if response.status_code != 200 or response.url == "https://www.marketbeat.com/stocks/NASDAQ/":
		return None;
	elif "dividend" not in response.url:
		# When the url has the wrong market name (e.g. NASDAQ instead of NYSE)
		# Marketbeat corrects that but doesn't move you
		# To the dividend page
		# Resend a request for the right exchange (from the redirect url)
		# But to the dividend page
		response = requests.get(response.url + "dividend/")

	# Parse the html
	soup = BeautifulSoup(response.content, "lxml")
	
	# Find the dividend stats section
	ticker_stats = soup.select("#cphPrimaryContent_tabDividendHistory .price-data")

	if ticker_stats:
		# Check for the frequency stat and get the text
		for stats_html in ticker_stats:
			text = stats_html.text
			if "Frequency" in text:
				print(ticker, "Available on Marketbeat")
				return stats_html.select("strong")[0].text
	else:
		ticker_stats = soup.select("#cphPrimaryContent_tabDividendHistory table tr")
		for stats_row in ticker_stats:
			text = stats_row.text
			if "Frequency" in text:
				try:
					return stats_row.select("td")[1].text
				except:
					continue
	
	return None

if __name__ == "__main__":
	# data_df = pd.read_csv("all_thestreet_dividends.csv")
	# data_df["Occurence"] = get_all_data(data_df["Symbol"])
	pass