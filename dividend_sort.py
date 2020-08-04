import pandas as pd
from datetime import date
from datetime import datetime, timedelta

entries = pd.read_csv("../../data/dividends/all_entries_filtered.csv")
entries.fillna('00000')

def all_tickers_today():
	# Note: this outputs the tickers for the next_day's ex-dividend date
	ex_dates_list = entries['Ex-Dividend Date']
	ex_ticker_list = entries['Ticker']
	today = str(date.today() + timedelta(days=1))
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
	today_ticker_dates_index = []
	for idx, item in ex_dates_list.iteritems():
		item = str(item)
		if item == today_date:
			today_ticker_dates_index.append(idx)
	ticker_list = []
	for i in today_ticker_dates_index:
		ticker_list.append(ex_ticker_list[i])
	return ticker_list, today_ticker_dates_index

def highest_paying_tickers():
	ticker_list, ticker_index = all_tickers_today()
	print(ticker_list)
	print(ticker_index)
	data = {'ticker': [],
			'payout': []}
	entries_today = pd.DataFrame(data, columns = ['ticker', 'payout'])
	for item in ticker_index:
		if entries['Period'][item] == 'monthly':
			payout = float(entries['Yield'][item][:-1]) / 12
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		if entries['Period'][item] == 'quarterly':
			payout = float(entries['Yield'][item][:-1]) / 4
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		if entries['Period'][item] == 'semi-annual':
			payout = float(entries['Yield'][item][:-1]) / 2
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		if entries['Period'][item] == 'annual':
			payout = float(entries['Yield'][item][:-1]) / 1
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
	# print(entries_today)
	entries_today = entries_today.nlargest(3, ['payout'])
	entries_today.reset_index(drop=True, inplace=True)
	print(entries_today)
	return entries_today




if __name__ == '__main__':
	highest_paying_tickers()
	# all_tickers_today()
# output ticker(s) to invest in for the day 