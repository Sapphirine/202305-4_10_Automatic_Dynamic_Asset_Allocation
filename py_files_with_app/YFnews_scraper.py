####################################################
#                                                  #
# Author(s): Dynamic Asset Allocation   - Zachary  #
#            Burpee, 							   #
# Class: Adv. Big Data Analytics Project           #
# Professor: Professor Ching-Yung Lin              #
# Description: Webscraping tool to scrape the      #
#              current news off of Yahoo Finance   #
#			   to determine sentiment              #
#                                                  #
####################################################

from bs4 import BeautifulSoup
import requests
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

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


def writeCSV(filename, l):
	''' Write to final CSV '''
	'''
	CSV_PATH = '/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/{}.csv'.format(filename)
	f = open(CSV_PATH, 'w')
	writer = csv.writer(f)
	header = ["News"]
	writer.writerow(header)
	f.close()
	'''
	
	CSV_PATH = '/Users/zacharyburpee/Desktop/Columbia/Spring 2023/Advanced Big Data & Ai/{}.csv'.format(filename)
	f = open(CSV_PATH, 'a')
	writer = csv.writer(f)
	for item in l:
		field = [item]
		writer.writerow(field)
	f.close()
	
	return

if __name__=="__main__":
	tokens = scrapeNews()
	model = keras.models.load_model('news_sentiment.h5')
	sequences = get_sequences(tokens)
	sentiments = model.predict(sequences)
	print(sentiments)
	for i in range(len(tokens)):
		 print(tokens[i], np.argmax(sentiments[i]), '\n')

	writeCSV('new_news', tokens)
