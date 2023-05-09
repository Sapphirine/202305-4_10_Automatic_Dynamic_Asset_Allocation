####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee,                               #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: App output                          #
#                                                  #
#                                                  #
#                                                  #
####################################################

from tkinter import *
from tkinter import ttk

from stock import load_dataset, trackNews
from LSTM import processData_LSTM, createModel_LSTM, predictNextWindow_LSTM
from DL import processData_DL, createModel_DL, predictNextWindow_DL
from RL import instant, training
import RL
from topLevel import portfolioWeights, calculateReturn, dailyUpdate, monthlyRetrain

import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import datetime as dt
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class App(Tk):
	def __init__(self):
		super().__init__()

		# configure the root window
		self.title('Automatic Dynamic Asset Allocation')
		self.geometry('1000x500')
		self.cur_row = 0
		self.cur_column = 0
		self.end_row = 0

		# Ticker & Weights - User
		self.tickers = dict()	# {"GC=F" : {"shares" : 4, "BOT" : $1922, "Current" : $1934, 
								# "Cost" : 4*1922, "weight" : cost/capital, "df": df, "vals": vals,
								# "lstm_20" : model.h5, "lstm_90" : model.h5, "dl" : model.h5}, ...}
		self.stopLoss = 0	# 10% selloff for stoploss
		self.capital = 0
		self.input = 1
		self.shares = []
		self.costs = []
		self.weights = []

		# label
		self.label = ttk.Label(self, text='Input Capital, Stop Loss, Current Shares & Prices')
		self.label.grid(row = self.cur_row, column = self.cur_column, columnspan = 3)
		self.cur_row += 1

		# Capital entry
		self.capital_label = ttk.Label(self, text='Input Capital')
		self.capital_label.grid(row = self.cur_row, column = 0)
		self.capital_box = ttk.Entry(self, background='white')
		self.capital_box.grid(row = self.cur_row, column = 1)
		self.cur_row += 1

		# Stop loss entry
		self.stl_label = ttk.Label(self, text='Stop Loss (Decimal)')
		self.stl_label.grid(row = self.cur_row, column = 0)
		self.stl_box = ttk.Entry(self, background='white')
		self.stl_box.grid(row = self.cur_row, column = 1)
		self.cur_row += 1

		# ticker entry
		self.entry_label = ttk.Label(self, text='Ticker')
		self.entry_label.grid(row = self.cur_row, column = 0)
		self.tic_entry = ttk.Entry(self, background='white')
		self.tic_entry.grid(row = self.cur_row, column = 1)
		self.cur_row += 1

		# shares entry
		self.shares_label = ttk.Label(self, text='Num. Shares')
		self.shares_label.grid(row = self.cur_row, column = 0)
		self.tic_shares = ttk.Entry(self, background='white')
		self.tic_shares.grid(row = self.cur_row, column = 1)
		self.cur_row += 1

		# BOT Price
		self.bot_label = ttk.Label(self, text='BOT Price')
		self.bot_label.grid(row = self.cur_row, column = 0)
		self.tic_bot = ttk.Entry(self, background='white')
		self.tic_bot.grid(row = self.cur_row, column = 1)
		self.cur_row += 1

		# ticker button
		self.button = ttk.Button(self, text='Add Ticker')
		self.button['command'] = self.add_ticker
		self.button.grid(row = self.cur_row, column = 0)

		# train models button
		self.button = ttk.Button(self, text='Train Models')
		self.button['command'] = self.train_models
		self.button.grid(row = self.cur_row, column = 1)

		# predict next day button
		self.button = ttk.Button(self, text='Balance Day')
		self.button['command'] = self.predict_day
		self.button.grid(row = self.cur_row, column = 2)
		self.cur_row += 1

		# Portfolio label
		self.labelp = ttk.Label(self, text='Portfolio Details:')
		self.labelp.grid(row = self.cur_row, column = 0)
		self.cur_row += 1

		# Gain Revenue
		self.total_val = ttk.Label(self, text="Current Value:")
		self.total_val.grid(row = 1, column=2,padx=50)
		self.gain = ttk.Label(self, text="{}".format(self.capital))
		self.gain.grid(row = 1, column = 3, padx=50)
		self.gain_percent = ttk.Label(self, text="{} %".format((self.capital-self.input)/self.input*100))
		self.gain_percent.grid(row=1, column=4, padx=50)

############################################################################################

	def add_ticker(self):
		#ent.delete(0, END) 
		# Add dictionary values
		self.capital = float(self.capital_box.get())
		self.input = float(self.capital_box.get())
		self.stopLoss = float(self.stl_box.get())

		# {"GC=F" : {"shares" : 4, "BOT" : $1922, "Current" : $1934, "Cost" : 4*1922, "weight" : cost/capital}
		entry = str(self.tic_entry.get()).upper()
		shares = int(self.tic_shares.get())
		self.shares.append(shares)
		bot = float(self.tic_bot.get())
		current = 0
		cost = shares * bot
		self.costs.append(cost)
		weight = cost / self.capital
		self.weights.append(weight)

		vals = dict()
		vals['shares'] = shares
		vals['BOT'] = bot
		vals['current'] = current
		vals['cost'] = cost
		vals['weight'] = weight

		# Append to tickers
		self.tickers[entry] = vals

		# Delete entries
		self.tic_entry.delete(0, END)
		self.tic_bot.delete(0, END)
		self.tic_shares.delete(0, END)

		# Print Known
		self.label2 = ttk.Label(self, text='Ticker={}'.format(entry))
		self.label2.grid(row = self.cur_row, column = 0, padx=50)
		self.label3 = ttk.Label(self, text='Shares={}'.format(shares))
		self.label3.grid(row = self.cur_row, column = 1, padx=50)
		self.label4 = ttk.Label(self, text='BOT={}'.format(bot))
		self.label4.grid(row = self.cur_row, column = 2, padx=50)
		self.label5 = ttk.Label(self, text='Cost={}'.format(cost))
		self.label5.grid(row = self.cur_row, column = 3, padx=50)
		self.label6 = ttk.Label(self, text='Weight={}'.format(weight))
		self.label6.grid(row = self.cur_row, column = 4, padx=50)
		self.cur_row += 1

		# indicate end row
		self.end_row = self.cur_row

	def train_models(self):
		# for each ticker
		for ticker in self.tickers:
			df = self.add_data(ticker)
			#exec(open("RL.py").read())
			rl_env = instant(df, ticker)
			training(rl_env)
			# 20 day LSTM
			lstm_data_20_x, lstm_data_20_y = processData_LSTM(df, 20)
			#lstm_20_model = createModel_LSTM(lstm_data_20_x, lstm_data_20_y, 20, ticker)
			self.tickers[ticker]['lstm_20'] = 'LSTM_20_{}.h5'.format(ticker)
			# 90 day LSTM
			lstm_data_90_x, lstm_data_90_y = processData_LSTM(df, 90)
			#lstm_90_model = createModel_LSTM(lstm_data_90_x, lstm_data_90_y, 90, ticker)
			self.tickers[ticker]['lstm_90'] = 'LSTM_90_{}.h5'.format(ticker)
			# DL 
			dl_data_x, dl_data_y = processData_DL(df)
			#dl_model = createModel_DL(dl_data_x, dl_data_y, ticker)
			self.tickers[ticker]['dl'] = 'DL_{}.h5'.format(ticker)
			# RL
			
			# vals dict - set all to neutral (HOLD)
			vals = dict()
			vals['sma30'] = 0
			vals['sma90'] = 0
			vals['sma200'] = 0
			vals['volume90'] = 0
			vals['rsi'] = 0
			vals['macd_trigger'] = 0
			vals['dl_predicted'] = 0
			vals['lstm_90_predicted'] = 0
			vals['lstm_20_predicted'] = 0
			self.tickers[ticker]['vals'] = vals
			self.tickers[ticker]['df'] = df

		return 1

	def add_data(self, ticker):
		df = load_dataset(ticker, "3y")
		return df

	def predict_day(self):
		new_values = []
		percents = []
		for ticker in self.tickers:
			vals = self.tickers[ticker]['vals']
			df = self.tickers[ticker]['df']
			new_vals, weight = dailyUpdate(ticker, df, vals)
			print(ticker, new_vals['close'])
			new_values.append(new_vals)
			percents.append(min(1, weight/9))

		# Update stocks
		capital = self.capital
		costs = self.costs
		weights = self.weights
		shares = self.shares
		outputs = calculateReturn(capital, costs, percents, weights, new_values, shares)
		# Display Results 
		self.displayTrades(outputs, new_values)
		# update new values
		i = 0
		for ticker in self.tickers:
			self.tickers[ticker]['vals'] = new_values[i]
			i += 1

	
	def displayTrades(self, outputs, vals):
		self.cur_row = self.end_row
		# Calculate new capital 
		# shares * close = capital
		self.capital = outputs[3]
		for i in range(len(vals)):
			self.capital += vals[i]['close'] * outputs[1][i] 

		# Pack capital
		self.total_val = ttk.Label(self, text="Current Value:")
		self.total_val.grid(row = 1, column=2,padx=50)
		self.gain = ttk.Label(self, text="{}".format(self.capital))
		self.gain.grid(row = 1, column = 3, padx=50)
		self.gain_percent = ttk.Label(self, text="{} %".format(round(((self.capital-self.input)/self.input*100), 2)))
		self.gain_percent.grid(row=1, column=4, padx=50)

		# Display trades
		keys = list(self.tickers.keys())
		# Pack trades:
		self.trade = ttk.Label(self, text="Daily Trades:")
		self.trade.grid(row = self.cur_row, column=0)
		self.cur_row += 1

		for i in range(len(self.shares)):
			if self.shares[i] < outputs[1][i]:
				diff = outputs[1][i] - self.shares[i]
				# Pack BUY
				self.buy = ttk.Label(self, text="BUY {} Shares of {}".format(diff, keys[i]))
				self.buy.grid(row = self.cur_row, column=1)
				self.cur_row += 1

			elif self.shares[i] > outputs[1][i]:
				# Pack SELL
				diff = self.shares[i] - outputs[1][i]
				self.sell = ttk.Label(self, text="SELL {} Shares of {}".format(diff, keys[i]))
				self.sell.grid(row = self.cur_row, column=1)
				self.cur_row += 1
				
			else:
				# Pack HOLD
				self.hold = ttk.Label(self, text="HOLD {} Shares of {}".format(self.shares[i], keys[i]))
				self.hold.grid(row = self.cur_row, column=1)
				self.cur_row += 1

if __name__ == "__main__":
	ENV_NAME = 'TradingEnv-v0'
	app = App()
	app.mainloop()
