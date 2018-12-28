#! /usr/bin/python3.5
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@nyu.edu'

import os
class Config():
	def __init__(self):
		self.model_name = 'att_bi_lstm_s2s'
		self.task = 'job640'
		self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
		self.TRAIN_PATH = os.path.join(self.CURR_PATH, 'data/{}/pp_data.json'.format(self.task))
		self.VOCAB_PATH = os.path.join(self.CURR_PATH, 'data/{}/vocab.json'.format(self.task))
		self.SAVE_POINT = os.path.join(self.CURR_PATH, 'log/{}/{}.ckpt'.format(self.model_name, self.model_name))
		self.train_percent = 0.9
		self.valid_save = True
		self.early_stop = 'acc'
		self.general_show = True
		self.sess_show = False
		self.train_show = True
		self.train_show_detail = False
		self.valid_show = True
		self.valid_show_detail = True
		self.test_show = True
		self.test_show_detail = True
		self.batch_size = 16
		self.train_step = 4
		self.valid_epoch = 1
		self.test_epoch = 256
		self.start_symbol = '<s>'
		self.end_symbol = '</s>'
		self.pad_symbol = '<pad>'
		self.init_scale = 0.08
		self.embedding_size = 128
		self.en_num_units = 128
		self.dense_activation = None
		self.dense_bias = False
		self.AttentionNormalize = False
		self.attention_num_units = 128
		self.attention_layer_size = 128
		self.de_num_units = 128
		self.en_drop_rate = 0.5
		self.de_drop_rate = 0.5
		self.beam_width = 10
		self.beam_len_norm = 0.0
		self.beam_conv_penalty = 0.0
		self.clipping_threshold = 5
		self.learning_rate = 0.001
		self.rmsp_decay_rate = 0.95