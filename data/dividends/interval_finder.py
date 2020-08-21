import pandas as pd
import numpy as np
import yfinance as yf

from scrape_ticker_stats import get_marketbeat_period
from cli_printing import print_progress_bar

entries_source = pd.read_csv("all_entries_filtered.csv")
entries_sink = pd.read_csv("all_thestreet_dividends.csv")

entries_by_ticker = entries_source.groupby("Ticker")

def interval_encoder():
	global entries_sink
	for idx_sink, line in entries_sink['Ticker'].iteritems():
		for idx_source, item in entries_source['Ticker'].iteritems():
			if line == item:
				period = entries_source['Period'][idx_source]
				entries_sink['Period'][idx_sink] = period
	entries_sink['Period'] = entries_sink['Period'].fillna('')
	for idx, item in entries_sink['Period'].iteritems():
		# entries_sink['Period'][idx] = str(item)
		# print(item)
		if item == '':
			entries_sink['Period'][idx] = 'quarterly'
			# print(item)

	entries_sink_new = entries_sink
	for idx, item in entries_sink['Ticker'].iteritems():
		stock = yf.Ticker(item)
		# print(item)
		history = stock.history(period = '1d')
		if history.empty:
			entries_sink_new = entries_sink_new.drop([idx])
			print(idx, '/', len(entries_sink['Ticker']))
		else:
			pass

	entries_sink = entries_sink_new[['Ticker', 'Ex-Dividend Date', 'Yield', 'Period']]
	entries_sink.to_csv(r"all_thestreet_dividends_updated.csv", index = False)

def get_period(ticker):
	try:
		return entries_source.loc[entries_source["Ticker"]==ticker, 'Period'].iloc[0]
	except:
		return None

def update_all_periods(entries_df):

	# Create a list for missing tickers, and one for periods
	missing_tickers = list()
	entries_df["Period"] = np.nan

	# Sort all entries by ticker to avoid
	# multiple lookups for the same ticker
	entries_df_by_ticker = entries_df.groupby("Ticker").groups

	# Pretty printing a progress bar
	print_data = {
		"fraction": True,
		"prefix": "Finding Dividend Periods:",
		"suffix": "Dates Checked",
		"total": entries_df.shape[0]
	}
	num_complete = 0
	print_progress_bar(num_complete, **print_data)

	# Iterate over tickers, not elements
	for ticker, indices in entries_df_by_ticker.items():
		
		# Try to get the period from other scraping tables
		# Otherwise scrape for it (much slower)
		period = get_period(ticker) or get_marketbeat_period(ticker)

		if period != None:
			entries_df.loc[indices, "Period"] = period
		else:
			missing_tickers.append(ticker)
		
		# Pretty printing
		num_complete += len(indices)
		print_progress_bar(num_complete, **print_data)

	print_progress_bar(num_complete, **print_data, printEnd="\n")
	return missing_tickers
		
def rename_periods(entries_df):
	replace_periods = {
		"Quarterly Dividend": "quarterly",
		"Monthly Dividend": "monthly",
		"Semi Annually Dividend": "semi-annual",
		"Annually Dividend": "annual",
		"N/A Dividend": np.nan,
		"--": np.nan,
		"none": np.nan,
		"aug 20": np.nan
	}
	entries_df["Period"].replace(replace_periods, inplace=True)

if __name__ == '__main__':
	update_all_periods(entries_sink)
	rename_periods(entries_sink)
	print(entries_sink.groupby("Period").count())
	entries_sink.to_csv(r"all_thestreet_dividends_updated.csv", index=False, na_rep="")