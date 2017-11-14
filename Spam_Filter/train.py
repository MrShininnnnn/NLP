TRAIN_HAM = "data/train_data/ham/" #0
TRAIN_SPAM = "data/train_data/spam/" #1

import os
import csv
import string
import numpy as np
import pandas as pd
import random
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

def load_data(path):

	data_set = []
	st_words = set(stopwords.words('english'))
	stemmer = LancasterStemmer()
	for file in os.listdir(path):
		with open(os.path.join(path,file), 'r', encoding = 'utf-8', errors = 'ignore') as f:
			pun_sent = []
			text = f.read()
			for letter in text:
				if letter not in string.punctuation:
					pun_sent.append(letter)
				else:
					pun_sent.append(" ")
			pun_sent = "".join(pun_sent)
			tokens = wordpunct_tokenize(pun_sent)
			data_set.append([stemmer.stem(token) for token in tokens if token not in st_words and len(token) != 1])
	
	return data_set

def word2list(data_set):
	
	all_words = []
	for sub_data_set in data_set:
		for sample in sub_data_set:
			for token in sample:
				all_words.append(token)

	word_count = Counter(all_words)
	word_list = [key for key, _ in word_count.most_common(20000)]

	return word_list

def word_embedding(data_set, word_list):

	embed_data_set = []
	for sample in data_set:
		sample_dict = dict((key, 0) for key in word_list)
		for token in sample:
			if token in word_list:
				sample_dict[token] += 1
		embed_data_set.append(np.fromiter(sample_dict.values(), dtype = float))

	return embed_data_set
	
def data_prepro(train_data_0, train_data_1):

	row_0 = len(train_data_0)
	row_1 = len(train_data_1)
	train_sample = np.asarray(train_data_0 + train_data_1)

	label_set_0 = [0] * row_0
	label_set_1 = [1] * row_1
	label_sample = np.asarray(label_set_0 + label_set_1)

	train_sample = TfidfTransformer().fit_transform(train_sample)

	return train_sample, label_sample

def main():

	print("Loading raw data set...")
	train_ham = load_data(TRAIN_HAM)
	train_spam = load_data(TRAIN_SPAM)

	print("Word embedding...")
	word_list = word2list([train_ham, train_spam])

	embed_train_ham = word_embedding(train_ham, word_list)
	embed_train_spam = word_embedding(train_spam, word_list)

	print("Preprocessing...")
	train_sample, train_label = data_prepro(embed_train_ham, embed_train_spam)

	print("Training...")
	clf = LinearSVC()
	cv = ShuffleSplit(n_splits = 5, test_size = 0.1)
	scores = cross_val_score(clf, train_sample, train_label, cv = cv)
	print("The accuracy rate on valid set is: %f" % (np.average(scores)))

if __name__ == "__main__":
	main()