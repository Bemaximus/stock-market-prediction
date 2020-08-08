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
	# today = str(date.today() - timedelta(days=(days_shift+1)))
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
		# 	stock = yf.Ticker(ex_ticker_list[idx])
		# 	history = stock.history(period = '1d')
		# 	if history.empty:
		# 		pass
		# 	else:
		# 		today_ticker_dates_index.append(idx)
			today_ticker_dates_index.append(idx)
	ticker_list = []
	# duplicate removal loop
	today_ticker_dates_index_new = []
	for i in today_ticker_dates_index:
		# ticker_list.append(ex_ticker_list[i])
		if ex_ticker_list[i] in ticker_list:
			pass
		else:
			ticker_list.append(ex_ticker_list[i])
			today_ticker_dates_index_new.append(i)
	# print(ticker_list)
	# print(len(ticker_list))
	# print(today_ticker_dates_index_new)
	# print(len(today_ticker_dates_index_new))
	return ticker_list, today_ticker_dates_index_new

def highest_paying_tickers(days_shift):
	ticker_list, ticker_index = all_tickers_today(days_shift)
	# print(ticker_list)
	# print(ticker_index)
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
	# print(entries_today)
	entries_today = entries_today.nlargest(3, ['payout'])
	entries_today.reset_index(drop=True, inplace=True)
	# print(entries_today)
	return entries_today

# def day_growth_calculator(len_in_days):
# 	day_growth = []
# 	for i in range(len_in_days):
# 		tickers = highest_paying_tickers(-i)
# 		tickers = tickers['ticker'].values.tolist()
# 		for item in tickers:
# 			stock = yf.Ticker(item)
# 			hist = stock.history(period=str(i+1)+'d')
# 			hist = hist.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
# 			day_growth.append(((hist['Close'][i]/hist['Open'][i])))
# 			print(hist)
# 		print(day_growth)
# 		print(tickers)

def stock_day_movement(ticker, date_, delta):
	stock = yf.Ticker(ticker)

	# delta = date.today() - date_
	# delta = int(str(delta).split(' ')[0])

	# print(delta)
	hist = stock.history(period=str(delta) + 'd')
	# print('er_history', hist)
	hist = hist.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
	day_movement = hist['Close'].iloc[0] / hist['Open'].iloc[0]
	# print('delta', delta)
	# print(hist['Close'])
	# print(hist['Close'][0])
	# print('es-date', str(hist.iloc[0]).split(' ')[-4])
	# print(hist)
	# print(day_movement)
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
	
	# stock = yf.Ticker(ticker)
	# delta = date.today() - date_before
	# # datetime.date(datetime.strptime(date_, '%m/%d/%Y'))
	# # print(delta)
	# delta = int(str(delta).split(' ')[0])
	# # print(delta)
	# hist = stock.history(period=str(delta) + 'd')
	# # print('er_history', hist)
	# hist = hist.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
	date_before = datetime.date(datetime.strptime(str(hist.iloc[0]).split(' ')[-4], '%Y-%m-%d'))
	# print(date_before)
	# print(ex_dates_list[value])
	return date_before

def backtest(how_far_back):
	movement_list = []
	for i in range(1, how_far_back):
		# print(i)
		tickers_today = highest_paying_tickers(i)
		# tickers = tickers_today['ticker'].values.tolist()
		temp_movement_list = []
		# print('temp ticker today', tickers_today)
		for item in tickers_today['ticker']:
			# ex_date = ex_date_from_ticker(item)
			ex_date = 0
			print('ticker_item', item) # undo this
			# print(ex_date)

			# print(ex_date)
			stock_day_move, ex_date = stock_day_movement(item, ex_date, i+1)
			# print('day move', stock_day_move)
			# if ex_date == 0:
			# 	print('error')
			# 	continue
			# stock_day_move, ex_date = stock_day_movement(item, ex_date)
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
	# highest_paying_tickers(0)
	# all_tickers_today()
	# day_growth_calculator(3)
	# stock_day_movement('SPY', '8/1/2020')
	# ex_date_from_ticker('NGL')
	backtest(100)
	# add_period()

"""
	TODO:
	-fix movement values not always accurate
	-account for weekends

"""
