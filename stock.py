####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee,                               #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: Data Processing                     #
#                                                  #
#                                                  #
#                                                  #
####################################################

import yfinance as yf
from bs4 import BeautifulSoup
import requests
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def downloader(ticker, time):
    history = yf.download(tickers = ticker,  # list of tickers
                    period = time,         # time period
                    interval = "1d",       # trading interval
                    prepost = False,       # download pre/post market hours data?
                    repair = True)
    return history

def SMA(df):
    # 200 SMA
    df['sma200'] = df['Close'].rolling(200).mean()
    # 90 SMA
    df['sma90'] = df['Close'].rolling(90).mean()
    # 30 SMA
    df['sma30'] = df['Close'].rolling(30).mean()

    return df

def MACD(df):
    k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    # Get the 12-day EMA of the closing price
    d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d
    # Get the 9-Day EMA of the MACD for the Trigger line
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    macd_h = macd - macd_s
    # Add all of our new values for the MACD to the dataframe
    df['macd'] = df.index.map(macd)
    df['macd_h'] = df.index.map(macd_h)
    df['macd_s'] = df.index.map(macd_s)

    return df

def RSI(df):
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    df['rsi'] = 100 - (100/(1 + rs))

    return df

def SMA_VOLUME(df):
    # 90 SMA
    df['volume90'] = df['Volume'].rolling(90).mean()

    return df

def load_dataset(ticker, time):
    df = downloader(ticker, time)
    # apply naming
    df['ticker'] = ticker
    # calculate SMAs
    SMA(df)
    # calculate MACD
    MACD(df)
    # calculate RSI
    RSI(df)
    # calculate SMA_Volume
    SMA_VOLUME(df)
    df = df.sort_values(by='Date')
    
    return df

def scrapeNews():
    '''
    Description: Funtion designed to gather all the news sources when called. 
    Inputs:
        -  
    Outputs:
        - Dictionary of current news sources by number and title w/ paragraph:
            {1: [title, summary], 2: [title2, summary2]}
    '''

    # Create custom URL to send to browser
    url = "https://finance.yahoo.com/news/"

    # Get HTML elements in text format
    html_content = requests.get(url).text

    # Parse the html content
    soup = BeautifulSoup(html_content, "lxml")
    # Return list of static news & ad results 
    all_results = soup.findAll("li", attrs={"class" : "js-stream-content Pos(r)"})

    tokens = []

    for i in range(len(all_results)):
        title_begin_index = str(all_results[i]).find('<u class="StretchedBox"></u>')
        title_substring = str(all_results[i])[title_begin_index:]
        title_end_index = title_substring.find('</a>')
        title = title_substring[28:title_end_index]

        summary_search = title_substring[title_end_index+12:]
        summary_begin_index = summary_search.find('">')
        summary_substring = summary_search[summary_begin_index:]
        summary_end_index = summary_substring.find('</p>')
        summary = summary_substring[2:summary_end_index]

        if len(title) > 0:
            tokens.append(title + " " + summary)

    return tokens

def get_sequences(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    #print("Vocab length", len(tokenizer.word_index) + 1)

    # Need to pad entirely on 53 since that is what the data was trained on
    sequences = pad_sequences(sequences, maxlen = 53, padding = 'post')

    return sequences


def trackNews(ticker):
	tokens = scrapeNews()
	model = keras.models.load_model('news_sentiment.h5')
	sequences = get_sequences(tokens)
	sentiments = model.predict(sequences)

	positives = 0
	mentions = 0
	for i in range(len(tokens)):
	# Check for stock mentions
		if ticker.lower() in tokens[i].lower():
			mentions += 1

			if np.argmax(sentiments[i]) == 0:
				positives += 1
			else:
				positives -= 1

	return positives, mentions



