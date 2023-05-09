####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee, 							   #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: Training Keras model for news       # 
#				sentiment model		               #
#                                                  #
####################################################

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from keras.metrics import Precision, Recall
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras import datasets

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

def get_sequences(texts):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)

	sequences = tokenizer.texts_to_sequences(texts)
	print("Vocab length", len(tokenizer.word_index) + 1)

	max_seq_length = np.max(list(map(lambda x: len(x), sequences)))
	print("Max sequence length", max_seq_length)

	sequences = pad_sequences(sequences, maxlen = max_seq_length, padding = 'post')

	return sequences


def preprocess_inputs(df):
	df = df.copy()

	sequences = get_sequences(df['Text'])

	label_mapping = {
		'negative': 0,
		'positive': 1
	}

	y = df['Label'].replace(label_mapping)

	train_sequences, test_sequences, y_train, y_test = train_test_split(sequences, y, train_size = 0.8, shuffle = True, random_state = 1)

	return train_sequences, test_sequences, y_train, y_test


if __name__=="__main__":
	data = pd.read_csv('/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/data.csv', \
					names=['Label', 'Text'], encoding='latin-1')

	X_train, X_val, y_train, y_val = preprocess_inputs(data)

	vocab_size = 45000
	embedding_size = 32
	epochs=30

	model= Sequential()
	model.add(Embedding(vocab_size, embedding_size, input_length=53))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Bidirectional(LSTM(32)))
	model.add(Dropout(0.4))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy', 
				optimizer='adam', 
               	metrics='accuracy')

	history = model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=8, epochs=epochs)


	results = model.evaluate(X_val, y_val)

	print("Test Losss: {:.5f}".format(results[0]))
	print("Test accuracy: {:.5f}%".format(results[1]  * 100))

	model.save("news_sentiment.h5")
	print("Saved Model")