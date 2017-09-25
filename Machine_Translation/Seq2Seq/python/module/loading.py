from module.path import *
from module import preprocess
import os

def sgm_check(sgm_file, str_file):

	if not os.path.isfile(str_file):
		if os.path.isfile(sgm_file):
			preprocess.sgm_to_str(sgm_file, str_file)
			print("Generate: " + str_file)
		else:
			print("Missing: " + sgm_file)

def load_data_and_vocab():

	sgm_check(train_en_sgm, train_en)
	sgm_check(train_zh_sgm, train_zh)
	sgm_check(valid_en_sgm, valid_en)
	sgm_check(valid_zh_sgm, valid_zh)
	sgm_check(test_en_sgm, test_en)
	