####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee,                               #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: DL Implementation                   #
#                                                  #
#                                                  #
#                                                  #
####################################################

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def processData_DL(df):

	df_plot = df.dropna()
	# reverse rows
	df_output = df_plot.iloc[::-1]
	data = df_output

	# eliminate ending one
	train = pd.DataFrame(data.iloc[:-1, :])
	train = train['Close'].values.reshape(-1, 1)

	# eliminate beginning one
	out = pd.DataFrame(data.iloc[1:, :])
	out = out['Close'].values.reshape(-1, 1)

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(train)
	scaled_out = scaler.fit_transform(out)


	x_train = scaled_data
	y_train = scaled_out

	return x_train, y_train

def createModel_DL(x_train, y_train, ticker):
	model = Sequential()
	model.add(Dense(64, input_shape = (1,1)))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mse')

	model.fit(x_train, 
          y_train, 
          epochs=50,
          batch_size = 8)

	model.save("DL_{}.h5".format(ticker))

	return model

def predictNextWindow_DL(model, test):
	df = test.dropna()
	# reverse rows
	df = df.iloc[::-1]

	#actual_prices = test_data.values
	scaler = MinMaxScaler(feature_range=(0,1))
	model_inputs = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

	x_test = model_inputs
	
	model.compile(optimizer='adam', loss='mse')
	predicted_prices = model.predict(x_test)
	prices = []
	for price in predicted_prices:
	    prices.append(price[0])

	prices = np.array(prices)
	predicted_prices = scaler.inverse_transform(prices)

	return predicted_prices












	