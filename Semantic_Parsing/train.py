#! /usr/bin/python3.5
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@nyu.edu'

import os
import random
import numpy as np
import tensorflow as tf
from src.source.config import Config
from src.toolkit.load import *
from src.toolkit.eva import *
from src.model.att_bi_lstm_s2s import  AttBiLSTM

class Parser():
	# a semantic parser for job640
	def __init__(self):

		print('\n**Model Initialization**')
		tf.reset_default_graph()
		self.config = Config()
		self.loadData()
		self.loadVocab()
		self.splitValid()
		self.setConfig()
		self.model = AttBiLSTM(self.config)
		self.setSessConfig()
		self.trainPrep()

	def loadData(self):

		self.data_dict = loadJson(self.config.TRAIN_PATH)
		self.train_dict = self.data_dict['train_dict']
		self.test_dict = self.data_dict['test_dict']

		self.train_en_x = np.array(self.train_dict['encoder_inputs'])
		self.train_de_x = np.array(self.train_dict['decoder_inputs'])
		self.train_de_y =np.array(self.train_dict['decoder_targets'])

		self.test_en_x = self.test_dict['encoder_inputs']
		self.test_de_x = self.test_dict['decoder_inputs']
		self.test_de_y = self.test_dict['decoder_targets']

	def loadVocab(self):

		self.vocab_dict = loadJson(self.config.VOCAB_PATH)
		self.all_vocab_dict = self.vocab_dict['all_vocab_dict']
		self.fl_vocab_dict = self.vocab_dict['fl_vocab_dict']

		self.all_index_dict = {v: k for k, v in self.all_vocab_dict.items()}
		self.fl_index_dict = {v: k for k, v in self.fl_vocab_dict.items()}

	def splitValid(self):

		index = np.random.permutation(len(self.train_en_x))
		index_list = np.split(index, [int(self.config.train_percent*len(self.train_en_x)), int(len(self.train_en_x))])
		
		self.train_index, self.valid_index = index_list[0], index_list[1]

		self.valid_en_x = self.train_en_x[self.valid_index]
		self.valid_de_x = self.train_de_x[self.valid_index]
		self.valid_de_y = self.train_de_y[self.valid_index]

		self.train_en_x = self.train_en_x[self.train_index]
		self.train_de_x = self.train_de_x[self.train_index]
		self.train_de_y = self.train_de_y[self.train_index]

	def setConfig(self):

		self.config.train_size = len(self.train_en_x)
		self.config.valid_size = len(self.valid_en_x)
		self.config.test_size = len(self.test_en_x)

		self.config.total_batch = self.config.train_size // self.config.batch_size + 1
		
		self.config.all_vocab_size = len(self.all_vocab_dict)
		self.config.fl_vocab_size = len(self.fl_vocab_dict)

		self.config.start_idx = self.all_vocab_dict[self.config.start_symbol]
		self.config.end_idx = self.all_vocab_dict[self.config.end_symbol]
		self.config.pad_idx = self.all_vocab_dict[self.config.pad_symbol]
		# decoder max time steps for inference only
		self.config.de_max_iter = max([len(tar) for tar in self.test_de_y])

	def setSessConfig(self):

		self.sess_config = tf.ConfigProto()
		self.sess_config.gpu_options.allow_growth = True
		self.init_l = tf.local_variables_initializer()
		self.init_g = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		if not self.config.sess_show:
			os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	def trainPrep(self):

		self.step = 0
		self.epoch = 0
		self.finished = False
		self.valid_loss = float('inf')
		self.valid_acc = float('-inf')
		self.valid_seq_acc = float('-inf')

	def trainShuffle(self):

		index_list = np.arange(self.config.train_size)
		np.random.shuffle(index_list)
		self.train_en_x = self.train_en_x[index_list]
		self.train_de_x = self.train_de_x[index_list]
		self.train_de_y = self.train_de_y[index_list]

	def trainer(self):

		self.sess = tf.Session(config=self.sess_config)
		self.sess.run(self.init_l)	
		self.sess.run(self.init_g)

		self.showInfo()

		while not self.finished:
			self.trainShuffle()
			for i in range(self.config.total_batch):
				# training
				train_en_x, train_de_x, train_de_y = self.getBatch(self.train_en_x, self.train_de_x, self.train_de_y, i)
				train_en_x, train_de_x, train_de_y = self.addSymbol(train_en_x, train_de_x, train_de_y)
				train_lens_x = self.getLen(train_en_x)
				train_lens_y = self.getLen(train_de_x)
				train_en_x, train_de_x, train_de_y= self.addPad(train_en_x, train_de_x, train_de_y)

				feed_dict = {
				self.model.x_inputs: train_en_x, 
				self.model.y_inputs: train_de_x,
				self.model.y_targets: train_de_y,
				self.model.en_lengths: train_lens_x, 
				self.model.de_lengths: train_lens_y, 
				self.model.en_drop_rate: self.config.en_drop_rate, 
				self.model.de_drop_rate: self.config.de_drop_rate}

				_, train_loss, train_acc, train_en_inputs, train_de_inputs, train_de_targets, train_de_preds = self.sess.run(
					[self.model.train_op, self.model.loss, self.model.acc, self.model.x_inputs, 
					self.model.y_inputs, self.model.y_targets, self.model.decoder_preds], 
					feed_dict=feed_dict)

				if self.step % self.config.train_step == 0 and self.step > 0:

					train_en_inputs, train_de_inputs, train_de_targets, train_de_preds = self.rmPad(
						train_lens_x, train_lens_y, train_en_inputs, train_de_inputs, train_de_targets, train_de_preds)


					train_seq_acc = Evaluate(train_de_targets, train_de_preds).acc()

					if self.config.train_show:
						
						print('Training epoch:{} step:{} loss:{}, acc:{}, seq_acc:{}'.format(
							self.epoch, self.step, train_loss, train_acc, train_seq_acc))

					if self.config.train_show_detail:
					
						en_input, de_input, de_tar, de_pred = self.randSample(
							train_en_inputs, train_de_inputs, train_de_targets, train_de_preds)
						
						print('en_input:', ' '.join(en_input))
						print('de_input:', ' '.join(de_input))
						print('de_target:', ' '.join(de_tar))
						print('de_pred:', ' '.join(de_pred), '\n')

				self.step += 1

			if self.epoch % self.config.valid_epoch == 0:

				valid_en_x, valid_de_x, valid_de_y = self.addSymbol(self.valid_en_x, self.valid_de_x, self.valid_de_y)
				valid_lens_x = self.getLen(valid_en_x)
				valid_lens_y = self.getLen(valid_de_x)
				valid_en_x, valid_de_x, valid_de_y= self.addPad(valid_en_x, valid_de_x, valid_de_y)

				feed_dict = {
				self.model.x_inputs: valid_en_x, 
				self.model.y_inputs: valid_de_x,
				self.model.y_targets: valid_de_y,
				self.model.en_lengths: valid_lens_x, 
				self.model.de_lengths: valid_lens_y}
				
				valid_loss, valid_acc, valid_en_inputs, valid_de_inputs, valid_de_targets, valid_de_preds = self.sess.run(
					[self.model.loss, self.model.acc, self.model.x_inputs, 
					self.model.y_inputs, self.model.y_targets, self.model.decoder_preds], 
					feed_dict=feed_dict)

				valid_en_inputs, valid_de_inputs, valid_de_targets, valid_de_preds = self.rmPad(
					valid_lens_x, valid_lens_y, valid_en_inputs, valid_de_inputs, valid_de_targets, valid_de_preds)


				valid_seq_acc = Evaluate(valid_de_targets, valid_de_preds).acc()

				self.earlyStop(valid_loss, valid_acc, valid_seq_acc)

				if self.config.valid_show:

					print('\nValidation epoch:{} step:{} loss:{}, acc:{}, seq_acc:{}'.format(
						self.epoch, self.step, valid_loss, valid_acc, valid_seq_acc))
					print('Validation best loss:{}, acc:{}, seq_acc:{}\n'.format(
						self.valid_loss, self.valid_acc, self.valid_seq_acc))

				if self.config.valid_show_detail:

					en_input, de_input, de_tar, de_pred = self.randSample(
						valid_en_inputs, valid_de_inputs, valid_de_targets, valid_de_preds)
					
					print('en_input:', ' '.join(en_input))
					print('de_input:', ' '.join(de_input))
					print('de_target:', ' '.join(de_tar))
					print('de_pred:', ' '.join(de_pred), '\n')

			if self.epoch >= self.config.test_epoch: 
				self.finished = True
			self.epoch += 1

	def addPad(self, en_inputs, de_inputs, de_targets):

		def seqPad(seq, length):
			return seq +[self.config.pad_idx] * (length - len(seq))

		en_max_len = max([len(seq) for seq in en_inputs])
		de_max_len = max([len(seq) for seq in de_inputs])

		pad_en_inputs = [seqPad(i, en_max_len) for i in en_inputs]
		pad_de_inputs = [seqPad(i, de_max_len) for i in de_inputs]
		pad_de_targets = [seqPad(i, de_max_len) for i in de_targets]

		return pad_en_inputs, pad_de_inputs, pad_de_targets

	def getBatch(self, en_inputs, de_inputs, de_targets, i):

		start_idx = i * self.config.batch_size
		end_idx = (i+1) * self.config.batch_size
		
		x_batch_inputs = en_inputs[start_idx: end_idx]
		y_batch_inputs = de_inputs[start_idx: end_idx]
		y_batch_targets = de_targets[start_idx: end_idx]

		return x_batch_inputs, y_batch_inputs, y_batch_targets

	def addSymbol(self, en_inputs, de_inputs, de_targets):

		ex_en_inputs = [seq + [self.config.end_idx] for seq in en_inputs]
		ex_de_targets = [seq + [self.config.end_idx] for seq in de_targets]
		ex_en_inputs = [[self.config.start_idx] + seq for seq in ex_en_inputs]
		ex_de_inputs = [[self.config.start_idx] + seq for seq in de_inputs]

		return ex_en_inputs, ex_de_inputs, ex_de_targets

	def getLen(self, seqs):

		return [len(seq) for seq in seqs]

	def rmPad(self, en_lengths, de_lengths, en_inputs, de_inputs, de_tars, de_preds):

		en_inputs = [en_inputs[i][:en_lengths[i]] for i in range(len(en_inputs))]
		de_inputs = [de_inputs[i][:de_lengths[i]] for i in range(len(de_inputs))]
		de_tars = [de_tars[i][:de_lengths[i]] for i in range(len(de_tars))]
		de_preds = [de_preds[i][:de_lengths[i]] for i in range(len(de_preds))]

		return en_inputs, de_inputs, de_tars, de_preds

	def randSample(self, en_inputs, de_inputs, de_tars, de_preds):

		index = random.sample(list(range(len(en_inputs))),1)[0]

		return self.index2Vcab(en_inputs[index], 'all'), \
		self.index2Vcab(de_inputs[index], 'fl'), \
		self.index2Vcab(de_tars[index], 'fl'), self.index2Vcab(de_preds[index], 'fl')

	def index2Vcab(self, seq, lan):

		if lan == 'all':
			return [self.all_index_dict[s] for s in seq]
		elif lan == 'fl':
			return [self.fl_index_dict[s] for s in seq]

	def showInfo(self):

		if self.config.general_show:
			print('\n*General Setting*')
			print('train sample size:', self.config.train_size)
			print('valid sample size:', self.config.valid_size)
			print('test sample size:', self.config.test_size)
			print('batch size:', self.config.batch_size)
			print('total batch:', self.config.total_batch)
			print('\n')

	def earlyStop(self, loss, acc, seq_acc):

		def validUpdate():
			
			self.valid_acc = acc
			self.valid_loss = loss
			self.valid_seq_acc = seq_acc

			if self.config.valid_save:
						self.saver.save(self.sess, self.config.SAVE_POINT)

		if self.config.early_stop == 'loss' and loss <= self.valid_loss: validUpdate()
		elif self.config.early_stop == 'acc' and acc >= self.valid_acc: validUpdate()
		elif self.config.early_stop == 'seq_acc' and seq_acc >= self.valid_seq_acc: validUpdate()

	def infer(self):

		self.saver.restore(self.sess, self.config.SAVE_POINT)

		test_en_x, test_de_x, test_de_y = self.addSymbol(self.test_en_x, self.test_de_x, self.test_de_y)

		test_lens_x = self.getLen(test_en_x)
		test_lens_y = self.getLen(test_de_x)
		test_en_x, test_de_x, test_de_y= self.addPad(test_en_x, test_de_x, test_de_y)

		test_feed_dict = {
		self.model.x_inputs: test_en_x, 
		self.model.y_inputs: test_de_x,
		self.model.y_targets: test_de_y,
		self.model.en_lengths: test_lens_x, 
		self.model.de_lengths: test_lens_y}

		tf_test_loss, tf_test_acc, tf_test_en_inputs, tf_test_de_inputs, tf_test_de_targets, tf_test_de_preds = self.sess.run(
			[self.model.loss, self.model.acc, self.model.x_inputs, 
			self.model.y_inputs, self.model.y_targets, self.model.decoder_preds], 
			feed_dict=test_feed_dict)

		tf_test_en_inputs, tf_test_de_inputs, tf_test_de_targets, tf_test_de_preds = self.rmPad(
			test_lens_x, test_lens_y, tf_test_en_inputs, tf_test_de_inputs, tf_test_de_targets, tf_test_de_preds)

		tf_test_seq_acc = Evaluate(tf_test_de_targets, tf_test_de_preds).acc()
		
		if self.config.test_show:
		
			print('\nTesting under teacher forcing:')
			print('loss:{}, acc:{}, seq_acc:{}'.format(tf_test_loss, tf_test_acc, tf_test_seq_acc))

		if self.config.test_show_detail:

			en_input, de_input, de_tar, de_pred = self.randSample(
				tf_test_en_inputs, tf_test_de_inputs, tf_test_de_targets, tf_test_de_preds)
				
			print('en_input:', ' '.join(en_input))
			print('de_input:', ' '.join(de_input))
			print('de_target:', ' '.join(de_tar))
			print('de_pred:', ' '.join(de_pred))

		inf_test_de_preds = self.sess.run(self.model.test_decoder_preds, feed_dict=test_feed_dict)
		inf_test_de_preds = [p[:len(t)] for p, t in zip(inf_test_de_preds, self.test_de_y)]

		inf_test_seq_acc = Evaluate(self.test_de_y, inf_test_de_preds).acc()
		
		if self.config.test_show:

			print('\nFinal Inference Result seq_acc:{}'.format(inf_test_seq_acc))

		if self.config.test_show_detail:

			index = random.sample(list(range(self.config.test_size)),1)[0]

			en_input = self.index2Vcab(self.test_en_x[index], 'all')
			de_tar = self.index2Vcab(self.test_de_y[index], 'fl')
			de_pred = self.index2Vcab(inf_test_de_preds[index], 'fl')

			print('source:', ' '.join(en_input))
			print('target:', ' '.join(de_tar))
			print('prediction:', ' '.join(de_pred), '\n')

def main():

	p = Parser()
	p.trainer()
	p.infer()

if __name__ == '__main__':
	main()
