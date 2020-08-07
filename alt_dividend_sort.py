import pandas as pd
from datetime import date
from datetime import datetime, timedelta
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) 

entries = pd.read_csv(dir_path + "/../../data/dividends/all_entries_filtered.csv")
entries = entries.dropna()

def all_tickers_today(current_date=date.today()):
	# Note: this outputs the tickers for the next_day's ex-dividend date
	
	today = str(current_date + timedelta(days=1))
	today_date = str(today[5:7]) + '/' + str(today[8:10]) + '/' + str(today[0:4])
	if today_date[0] == '0':
		today_date_month = today_date[1]
	else:
		today_date_month = today_date[0:2]
	if today_date[3] == '0':
		today_date_day = today_date[4]
	else:
		today_date_day = today_date[3:5]
	today_date_year = today_date[6:]
	today_date = today_date_month + '/' + today_date_day + '/' + today_date_year
	
	ex_dates_list = entries['Ex-Dividend Date']
	ex_ticker_list = entries['Ticker']
	today_entries = entries[entries['Ex-Dividend Date'] == today_date].drop_duplicates('Ticker')
	
	# today_ticker_dates_index = []
	# for idx, item in ex_dates_list.iteritems():
	# 	item = str(item)
	# 	if item == today_date:
	# 		today_ticker_dates_index.append(idx)
	# ticker_list = []
	# # duplicate removal loop
	# for i in today_ticker_dates_index:
	# 	if ex_ticker_list[i] in ticker_list:
	# 		today_ticker_dates_index.remove(i)
	# 	else:
	# 		ticker_list.append(ex_ticker_list[i])
	# print(ticker_list)
	# print(today_ticker_dates_index)
	return today_entries

def highest_paying_tickers(num_tickers=2):
	today_tickers = all_tickers_today()
	# print(ticker_list)
	# print(ticker_index)
	data = {'ticker': [],
			'payout': []}
	entries_today = pd.DataFrame(data, columns = ['ticker', 'payout'])
	for i, (ticker, date, perc_yield, period) in today_tickers.iterrows():
		if period == 'monthly':
			payout = float(perc_yield[:-1]) / 12
			new_row = {'ticker': ticker, 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		if period == 'quarterly':
			payout = float(perc_yield[:-1]) / 4
			new_row = {'ticker': ticker, 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		if period == 'semi-annual':
			payout = float(perc_yield[:-1]) / 2
			new_row = {'ticker': ticker, 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		if period == 'annual':
			payout = float(perc_yield[:-1]) / 1
			new_row = {'ticker': ticker, 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
	# print(entries_today)
	entries_today = entries_today.nlargest(num_tickers, ['payout'])
	entries_today.reset_index(drop=True, inplace=True)
	print(entries_today)
	return entries_today




if __name__ == '__main__':
	highest_paying_tickers()
	# all_tickers_today()
# output ticker(s) to invest in for the day 