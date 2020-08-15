import os
import pandas as pd

def download_ticker_headlines(ticker, company_name, number_of_messages=10000, custom_name="tweets"):
	"""
	Call the twitterscraper command to download tweets

	"""
	exit_code = os.system(f'twitterscraper "${ticker} OR {company_name} from:wsj' +\
		f' OR from:reuters OR from:business OR from:cnbc ' +\
		f'OR from:RANsquawk OR from:wsjmarkets" ' +\
		f'-o twitterscraper_{ticker}_{custom_name}.json -l {number_of_messages}')
	return exit_code

def get_ticker_headlines(ticker, company_name=None, custom_filename=None):
	"""
	return a pandas dataframe of tweets
	"""
	try:
		if custom_filename:
			return pd.read_json(custom_filename, encoding='utf-8')
		else:
			return pd.read_json(f"twitterscraper_{ticker}_tweets.json", encoding='utf-8')
	except Exception as e:
		download_ticker_headlines(ticker, company_name if company_name else ticker)
		return pd.read_json(f"twitterscraper_{ticker}_tweets.json", encoding='utf-8')

if __name__ == "__main__":
	df = get_ticker_headlines("TSLA", "Tesla")