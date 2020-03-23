# Copyright (c) 2018 Benson Tran, MIT License

import requests
import pandas as pd
import arrow
import datetime
import pandas as pd
import numpy as np


def get_quote_data(symbol='AAPL', data_range='7d', data_interval='1m'):
    res = requests.get(
        f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}' )
    data = res.json()
    body = data['chart']['result'][0]
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='dt')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])
    df = df.loc[:, ('close', 'volume')]
    df.dropna(inplace=True)  # removing NaN rows
    df.columns = ['Price', 'Volume']  # Renaming columns in pandas

    start_date = df.index[0].strftime('%Y%m%d')
    out_filename = f"{symbol}{start_date}{data_range}{data_interval}.csv"
    df.to_csv(out_filename)


    return df

if __name__ == "__main__":
    data = get_quote_data(input('ticker (ex "JNUG"): '),
        input('range (ex "7d"): '), input('interval (ex "1m"): '))
    print(data)
