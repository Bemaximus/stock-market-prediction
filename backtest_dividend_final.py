import pandas as pd
from datetime import date
from datetime import datetime, timedelta
import yfinance as yf
import numpy
import timeboard as tb
import timeboard.calendars.US as US

entries = pd.read_csv("../../data/dividends/all_entries_filtered.csv")
entries = pd.read_csv("../../data/dividends/all_thestreet_dividends_updated.csv")
# entries = pd.read_csv("../../data/dividends/all_entries_filtered_andrew.csv")
entries = entries.dropna()
print(entries)

def all_tickers_today(days_shift):
	# Note: this outputs the tickers for the next_day's ex-dividend date
	stock = yf.Ticker('SPY')
	stock_hist = stock.history(period = str(days_shift) + 'd')
	today = str(datetime.date(datetime.strptime(str(stock_hist.iloc[0]).split(' ')[-4], '%Y-%m-%d')))
	ex_dates_list = entries['Ex-Dividend Date']
	ex_ticker_list = entries['Ticker']
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

	# duplicate removal loop
	today_ticker_dates_index_new = []
	for i in today_ticker_dates_index:
		if ex_ticker_list[i] in ticker_list:
			pass
		else:
			ticker_list.append(ex_ticker_list[i])
			today_ticker_dates_index_new.append(i)
	return ticker_list, today_ticker_dates_index_new

def highest_paying_tickers(days_shift):
	ticker_list, ticker_index = all_tickers_today(days_shift)
	data = {'ticker': [],
			'payout': []}
	entries_today = pd.DataFrame(data, columns = ['ticker', 'payout'])
	for item in ticker_index:
		if entries['Period'][item] == 'monthly':
			payout = float(entries['Yield'][item][:-1]) / 12
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		elif entries['Period'][item] == 'quarterly':
			payout = float(entries['Yield'][item][:-1]) / 4
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		elif entries['Period'][item] == 'semi-annual':
			payout = float(entries['Yield'][item][:-1]) / 2
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
		elif entries['Period'][item] == 'annual':
			payout = float(entries['Yield'][item][:-1]) / 1
			new_row = {'ticker': entries['Ticker'][item], 'payout': payout}
			entries_today = entries_today.append(new_row, ignore_index=True)
	entries_today = entries_today.nlargest(3, ['payout'])
	entries_today.reset_index(drop=True, inplace=True)
	return entries_today

def stock_day_movement(ticker, delta):
	stock = yf.Ticker(ticker)
	hist = stock.history(period=str(delta) + 'd')
	hist = hist.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
	day_movement = hist['Close'].iloc[0] / hist['Open'].iloc[0]
	return day_movement, str(hist.iloc[0]).split(' ')[-4]

def ex_date_from_ticker(ticker):
	# gets date 1 day before ex-date
	ex_ticker_list = entries['Ticker']
	for idx, item in ex_ticker_list.iteritems():
		item = str(item)
		if item == ticker:
			value = idx
	ex_dates_list = entries['Ex-Dividend Date']
	date_before = datetime.date(datetime.strptime(ex_dates_list[value], '%m/%d/%Y')) - timedelta(days=1)
	date_before = datetime.date(datetime.strptime(str(hist.iloc[0]).split(' ')[-4], '%Y-%m-%d'))
	return date_before

def backtest(how_far_back):
	movement_list = []
	for i in range(1, how_far_back):
		tickers_today = highest_paying_tickers(i)
		temp_movement_list = []
		for item in tickers_today['ticker']:
			print('ticker_item', item) # undo this
			stock_day_move, ex_date = stock_day_movement(item, i+1)
			print(ex_date) # undo this
			temp_movement_list.append(stock_day_move)
		print('temp_movement_list', temp_movement_list) # undo this
		if len(temp_movement_list) == 0:
			pass
		else:
			portfolio_daily_move = sum(temp_movement_list) / len(temp_movement_list)
			movement_list.append(portfolio_daily_move)
	print(movement_list)
	total_movement = numpy.prod(movement_list)
	print('portfolio movement:', total_movement)
	return total_movement


if __name__ == '__main__':
	backtest(130)
