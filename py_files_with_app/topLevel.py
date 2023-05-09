####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee, 							   #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: topLevel code for helper functions  #
#              in app.py   						   #
#			               						   #
#                                                  #
####################################################

import stock
from LSTM import predictNextWindow_LSTM
from DL import predictNextWindow_DL
import RL
import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import datetime as dt
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import math


def portfolioWeights(tickers, dfs, vals):
	percent = [i for i in range(len(tickers))]
	for i in range(len(tickers)):
		new_values, weight = dailyUpdate(tickers[i], dfs[i], vals[i])
		percent[i] = min(1, weight/9)

	return percent


def calculateReturn(capital, costs, percents, weights, vals, shares):
	selloff = 0
	for i in range(len(vals)):
		percent = percents[i] + 1
		# Buy more
		if percent >= 1:
			weight = weights[i] * percent
			share = int(shares[i]*weight)
			costs[i] += abs((share-shares[i])) * vals[i]['close']

			# rain check
			if sum(costs) > capital:
				diff = abs(sum(costs) - capital)
				diff_s = math.ceil(diff/(costs[i]/share))
				costs[i] -= diff

		# selloff
		elif percent < 1:
			weight = weights[i] * percent
			share = int(shares[i]*weight)
			costs[i] -= abs((shares[i]-share)) * vals[i]['close']
			selloff += abs((shares[i]-share)) * vals[i]['close']
			# rain check
			if costs[i] < 0:
				shares[i] = 0
				costs[i] = 0

	return costs, shares, weights, selloff


def dailyUpdate(ticker, df, vals):
	values = dict()		# {'close': , ..}
	weight = 0

	history = yf.download(tickers = ticker,  # list of tickers
            period = "200d",         # time period
            interval = "1d",       # trading interval
            prepost = False,       # download pre/post market hours data?
            repair = True)

	current = float(history['Close'][-1])

	# current val
	values['close'] = current

	# sma30
	if df['sma30'][-1] > current and vals['sma30'] != 1:
		values['sma30'] = 1
		weight += 1
	elif df['sma30'][-1] < current and vals['sma30'] != -1:
		values['sma30'] = -1
		weight -= 1
	else:
		values['sma30'] = 0

	# sma90
	if df['sma90'][-1] > current and vals['sma90'] != 1:
		values['sma90'] = 1
		weight += 1
	elif df['sma90'][-1] < current and vals['sma90'] != -1:
		values['sma90'] = -1
		weight -= 1
	else:
		values['sma90'] = 0

	# sma200
	if df['sma200'][-1] > current and vals['sma200'] != 1:
		values['sma200'] = 1
		weight += 1
	elif df['sma200'][-1] < current and vals['sma200'] != -1:
		values['sma200'] = -1
		weight -= 1
	else:
		values['sma200'] = 0

	# sma Volume
	if df['volume90'][-1] > current and vals['volume90'] != 1:
		values['volume90'] = 1
		weight += 1
	elif df['volume90'][-1] < current and vals['volume90'] != -1:
		values['volume90'] = -1
		weight -= 1
	else:
		values['volume90'] = 0

	# RSI
	if df['rsi'][-1] > 70 and vals['rsi'] != 1:
		values['rsi'] = 1
		weight += 1
	elif df['rsi'][-1] < 30 and vals['rsi'] != -1:
		values['rsi'] = -1
		weight -= 1
	else:
		values['rsi'] = 0

	# macD
	if df['macd'][-1] > df['macd_s'][-1] and vals['macd_trigger'] != 1:
		values['macd_trigger'] = 1
		weight += 1
	elif df['macd'][-1] < df['macd_s'][-1] and vals['macd_trigger'] != -1:
		values['macd_trigger'] = -1
		weight -= 1
	else:
		values['macd_trigger'] = 0

	# news
	values['news_mention'], values['news_positive'] = stock.trackNews(ticker)

	# predicted models
	dl_model = load_model('/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/DL_{}.h5'.format(ticker))
	lstm_90_model = load_model('/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/LSTM_90_{}.h5'.format(ticker))
	lstm_20_model = load_model('/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/LSTM_20_{}.h5'.format(ticker))

	dl_predicted = predictNextWindow_DL(dl_model, df)[-1]
	lstm_90_predicted = predictNextWindow_LSTM(lstm_90_model, history, df, 90)[-1]
	lstm_20_predicted = predictNextWindow_LSTM(lstm_20_model, history, df, 20)[-1]

	# Check models
	if dl_predicted > current and vals['dl_predicted'] != -1:
		values['dl_predicted'] = -1
		weight -= 2
	elif dl_predicted < current and vals['dl_predicted'] != 1:
		values['dl_predicted'] = 1
		weight += 2
	else:
		values['dl_predicted'] = 0

	if lstm_90_predicted > current and vals['lstm_90_predicted'] != -1:
		values['lstm_90_predicted'] = -1
		weight -= 2
	elif lstm_90_predicted < current and vals['lstm_90_predicted'] != 1:
		values['lstm_90_predicted'] = 1
		weight += 2
	else:
		values['lstm_90_predicted'] = 0

	if lstm_20_predicted > current and vals['lstm_20_predicted'] != -1:
		values['lstm_20_predicted'] = -1
		weight -= 2
	elif lstm_20_predicted < current and vals['lstm_20_predicted'] != 1:
		values['lstm_20_predicted'] = 1
		weight += 2
	else:
		values['lstm_20_predicted'] = 0

	return values, weight

def monthlyRetrain(df, ticker):
	x_90, y_90 = LSTM.processData(df, 90)
	model_90 = LSTM.createModel(x_90, y_90, 90, ticker)
	
	x_20, y_20 = LSTM.processData(df, 20)
	model_20 = LSTM.createModel(x_20, y_20, 20, ticker)

	x, y = DL.processData(df)
	model = LSTM.createModel(x, y, ticker)

	return 1

	

