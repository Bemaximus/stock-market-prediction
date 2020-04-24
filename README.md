# Stock Market Prediction
Predicting the stock market using MATLAB and linear regression

Using the opening price, high price, and low price of a stock over two weeks, as well as the current stock opening price, this model predicts the closing price of that stock with about 80% accuracy.

![](https://github.com/intermezzio/stock-market-prediction/blob/matlab_original_algorithm/output/jnug_gains.png)

[Full Report Here](https://docs.google.com/document/d/1L3u5gKNvpuLp4S4-yjCEbRjHrwH91EEyRn_v-18-4_s/edit?usp=sharing)

## Repository Structure
The `data/` folder stores csv files in the form \<TICKER\>.csv which include historical data for each stock.\
This folder also has a shell script to import this from Yahoo Finance (Copyright (c) 2017 Brad Lucas, MIT License) \
The `minute/` subfolder also has a limited amount of minute data
  
The `backtesting-python/` folder uses backtesting libraries to test algorithms over time

The `output/` folder stores output data in text files

The `strategies/` directory stores worktrees (branches as folders) for each strategy implemented

The `live_trading/` directory live trades the algorithm using the Alpaca API

## Usage

To set up the repository, run `sh setup.sh` in the main folder of the repository. Then, if you want to trade on the market, add your API keys to the `live_trading/config/config.py` folder. \
To run the trading algorithm 24/7, run `nohup python3 live_trading/alpaca_sdk_trade.py auto`. You may want to run this on a virtual machine that stays online forever for optimal results.
