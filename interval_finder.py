import pandas as pd
import yfinance as yf

entries_source = pd.read_csv("../../data/dividends/all_entries_filtered.csv")
entries_sink = pd.read_csv("../../data/dividends/all_thestreet_dividends.csv")

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
	entries_sink.to_csv(r"../../data/dividends/all_thestreet_dividends_updated.csv", index = False)



if __name__ == '__main__':
	interval_encoder()