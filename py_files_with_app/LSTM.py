####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee,                               #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: LSTM Implementation                 #
#                                                  #
#                                                  #
#                                                  #
####################################################

import numpy as np
import pandas as pd
import os
import stock
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

'''
90 day is long term, 20 day is good
'''

def processData_LSTM(df, days):
	df_plot = df.dropna()
	# reverse rows
	df_output = df_plot.iloc[::-1]
	#df_output = df_output.drop('Date', axis=1)
	data = df_output

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
	# how many days do i want to base my predictions on ?
	prediction_days = days

	x_train = []
	y_train = []

	for x in range(prediction_days, len(scaled_data)):
		x_train.append(scaled_data[x - prediction_days:x, 0])
		y_train.append(scaled_data[x, 0])
	    
	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	return x_train, y_train

def createModel_LSTM(x_train, y_train, days, ticker):
	model = Sequential()
	model.add(LSTM(64, return_sequences = True, input_shape = (x_train.shape[1],1)))
	model.add(Dropout(0.1))
	model.add(LSTM(64, return_sequences = True))
	model.add(Dropout(0.1))
	model.add(LSTM(32))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mse', metrics='mse')

	model.fit(x_train, 
	          y_train, 
	          epochs=100,
	          batch_size = 16)

	model.save("LSTM_{}_{}.h5".format(days, ticker))

	return model

def testData(model, days):
	df = pd.read_csv('/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/test.csv')
	df = df.dropna()
	# reverse rows
	df = df.iloc[::-1]
	df = df.drop('date', axis=1)
	test_data = df.astype('float')

	actual_prices = test_data['close'].values

	total_dataset = pd.concat((data['close'], test_data['close']), axis=0)

	model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
	model_inputs = model_inputs.reshape(-1,1)
	model_inputs = scaler.fit_transform(model_inputs)


	x_test = []
	for x in range(prediction_days, len(model_inputs)):
	    x_test.append(model_inputs[x-prediction_days:x, 0])

	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

	predicted_prices = model.predict(x_test)
	predicted_prices = scaler.inverse_transform(predicted_prices)

	# 90 day 100 epochs
	plt.plot(actual_prices, color='black', label="Actual price")
	plt.plot(predicted_prices, color= 'green', label="predicted price")
	plt.title("share price")
	plt.xlabel("time")
	plt.ylabel("share price")
	plt.legend()
	plt.show()

def predictNextWindow_LSTM(model, original, test, prediction_days):
	df = test.dropna()
	# reverse rows
	df = df.iloc[::-1]

	total_dataset = pd.concat((original['Close'], df['Close']), axis=0)
	scaler = MinMaxScaler(feature_range=(0,1))
	model_inputs = total_dataset[len(total_dataset) - len(df) - prediction_days:].values
	model_inputs = model_inputs.reshape(-1,1)
	model_inputs = scaler.fit_transform(model_inputs)


	x_test = []
	for x in range(prediction_days, len(model_inputs)):
	    x_test.append(model_inputs[x-prediction_days:x, 0])

	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

	model.compile(optimizer='adam', loss='mse', metrics='mse')
	predicted_prices = model.predict(x_test)
	predicted_prices = scaler.inverse_transform(predicted_prices)

	return predicted_prices




