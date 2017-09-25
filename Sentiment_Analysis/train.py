source_train_path = "data/aclImdb/train/" # train data
train_path = "data/train/"
train_data = train_path + "imdb_tr.csv" # train data after preprocessing

source_test_path = "data/aclImdb/test/" # test data
test_path = "data/test/"
test_data = test_path + "imdb_te.csv" # test data after preprocessing

output_path = "output/" # to store predictde results
uni_output = output_path + "unigram.output.txt"
unigramtfidf_output = output_path + "unigramtfidf.output.txt"
bi_output = output_path + "bigram.output.txt"
bigramtfidf_output = output_path + "bigramtfidf.output.txt"

import os
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

def preprocess_words(words):
	# words tokenization
	words = words.lower()
	no_punct = ""
	stop_words = stopwords.words("english")
	for  w in words:
		if w not in string.punctuation:
			no_punct = no_punct + w
		elif w in ''' ' ''':
			no_punct = no_punct + " "
	no_punct_words = no_punct.split()
	resultwords = [word for word in no_punct_words if word not in stop_words]
	result = ' '.join(resultwords)
	return result


def imdb_data_preprocess(inpath, outpath="./", name=[train_data, test_data], path = [train_path, test_path]):
	# split source data into three set
	if os.path.isfile(name[0]) and os.path.isfile(name[1]):
		return
	rawdata = []
	c = 0
	for dir_name in ("pos/", "neg/"):
		if dir_name == "pos/":
			label = 1
		else:
			label = 0
		txt_dir = os.listdir(inpath + dir_name)
		for txt in txt_dir:
			file = open(inpath + dir_name + txt, "r")
			words = file.read()
			words = preprocess_words(words)
			rawdata.append((c, words, label))
			c += 1
	rawdata_frame = pd.DataFrame(rawdata, columns = ["row_number", "text", "polarity"])
	for folder in path:
		if not os.path.exists(folder):
			os.makedirs(folder)
	if inpath == source_train_path:
		rawdata_frame.to_csv(name[0], index = False)
	else:
		rawdata_frame.to_csv(name[1], index = False)

def data_loading(train_data, test_data):
	#load samples and labels from data csv
	train_file = pd.read_csv(train_data)
	test_file = pd.read_csv(test_data)
	return train_file["text"].values, train_file["polarity"].values, test_file["text"].values, test_file["polarity"].values

def unigram_sgd(train_text, train_label, test_text, test_label):
	# unigram
	cv = CountVectorizer()
	train_v = cv.fit_transform(train_text)
	test_v = cv.transform(test_text)
	clf = SGDClassifier(loss = "hinge", penalty = "l1").fit(train_v, train_label)
	result_unigram = clf.predict(test_v)
	output_txt(result_unigram, uni_output)
	score = evaluate(result_unigram, test_label)
	print("The score based on unigram:", score, "%")
	# unigram_tfidf
	tf = TfidfTransformer()
	train_v = tf.fit_transform(train_v)
	test_v = tf.transform(test_v)
	clf = SGDClassifier(loss = "hinge", penalty = "l1").fit(train_v, train_label)
	result_unigram_tfidf = clf.predict(test_v)
	output_txt(result_unigram_tfidf, unigramtfidf_output)
	score = evaluate(result_unigram_tfidf, test_label)
	print("The score based on unigram_tfidf:", score, "%")

def bigram_sgd(train_text, train_label, test_text, test_label):
	# bigram
	bcv = CountVectorizer(ngram_range = (1, 2))
	train_v = bcv.fit_transform(train_text)
	test_v = bcv.transform(test_text)
	clf = SGDClassifier(loss = "hinge", penalty = "l1").fit(train_v, train_label)
	result_bigram = clf.predict(test_v)
	output_txt(result_bigram, bi_output)
	score = evaluate(result_bigram, test_label)
	print("The score based on bigram:", score, "%")
	# bigram_tfidf
	tf = TfidfTransformer()
	train_v = tf.fit_transform(train_v)
	test_v = tf.transform(test_v)
	clf = SGDClassifier(loss = "hinge", penalty = "l1").fit(train_v, train_label)
	result_bigram_tfidf = clf.predict(test_v)
	output_txt(result_bigram_tfidf, bigramtfidf_output)
	score = evaluate(result_bigram_tfidf, test_label)
	print("The score based on bigram_tfidf:", score, "%")

def output_txt(result, txt_name):

	file = open(txt_name, "w")
	for num in result:
		file.write("%s\n" % num)
	file.close()

def evaluate(predict, test_label):

	size = predict.shape[0]
	score = (1 - np.sum(np.absolute(predict - test_label))/size) * 100
	return score

if __name__ == "__main__":

	print("Data preprocessing...")
	for path in [source_train_path, source_test_path]:
		imdb_data_preprocess(path)

	print("Data loading...")
	train_text, train_label, test_text, test_label = data_loading(train_data, test_data)

	print("Training...")
	if not os.path.exists(output_path): os.makedirs(output_path)
	unigram_sgd(train_text, train_label, test_text, test_label)
	bigram_sgd(train_text, train_label, test_text, test_label)
