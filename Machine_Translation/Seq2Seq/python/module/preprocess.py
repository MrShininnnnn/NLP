import os
import re
import json
from collections import Counter
from tqdm import tqdm
import jieba
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup, SoupStrainer
from module.path import *

def sgm_to_str(sgm_file, str_file):
	
	raw_data = open(sgm_file, 'r').read()
	f = open(str_file, 'w+')
	soup = BeautifulSoup(raw_data, 'lxml')
	for sample in soup.find_all('seg'):
		sen = sample.get_text()
		f.write(sen.strip() + "\n")
	f.close()

def str_to_dict(str_file, dict_file, lan):

	PREFIX_VOCAB = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
	vocab_path = en_dict
	vocab_counter = Counter()
	with open(str_file, 'r', encoding = 'utf-8') as infile:
		for line in tqdm(infile, total = os.path.getsize(str_file)):
			tokens = word_tokenize(line)
			vocab_counter.update(tokens)
	vocab_list = [key for key, value in en_vocab_counter.items() if value > 0]
	vocab_list = PREFIX_VOCAB + vocab_list
	save_vocabulary(vocab_list, dict_file)

def load_vocabulary(vocab_path):
    print("loading {}".format(vocab_path))
    vocab_list = []
    with open(vocab_path, 'r', encoding='utf-8') as fh: 
        for line in tqdm(fh):
            ls = line.rstrip().split("\t")
            vocab_list.append(ls[0])

    return vocab_list

def save_vocabulary(vocab_list, vocab_path):
    with open(vocab_path, 'w', encoding='utf-8') as fh:
        for idx, vocab in enumerate(vocab_list):
            fh.write("{}\t{}\n".format(vocab, idx))
