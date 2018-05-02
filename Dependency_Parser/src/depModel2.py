#! /usr/bin/python3.5

WORD = './data/vocabs.word'
POS = './data/vocabs.pos'
LABELS = './data/vocabs.labels'
ACTIONS = './data/vocabs.actions'
TRAIN = './data/train.data'
DEV = './data/dev.data'
MODEL = './src/tmp/depModel2/model.ckpt'
CHECK = './src/tmp/depModel2/checkpoint'
GRPAH = './src/tmp/depModel2/model.ckpt.meta'

import os,sys
from utils import *
from decoder import *
from configuration import Configuration
import numpy as np
import tensorflow as tf
import time

class depModel():

	def __init__(self):
		'''
		You can add more arguments for examples actions and model paths.
		You need to load your model here.
		actions: provides indices for actions.
		it has the same order as the data/vocabs.actions file.
		'''
		# if you prefer to have your own index for actions, change this.
		self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']
		# write your code here for additional parameters.
		# feel free to add more arguments to the initializer.
		self.vocabsWord = self.vocabsWord()
		self.vocabsPos = self.vocabsPos()
		self.vocabsLabels = self.vocabsLabels()
		self.vocabsActions = self.vocabsActions()
		self.trainSet = self.trainSet()
		self.devSet = self.devSet()
		self.modelGraph = self.modelGraph()
		self.sess = tf.Session()
		self.modelTrained = self.modelTrained()
		self.i = 0

	def vocabsWord(self):
		
		word_dict = dict()
		with open(WORD) as file:
			for line in file:
				line = line.strip().split()
				word_dict[line[0]] = int(line[1])
		
		return word_dict

	def vocabsPos(self):
		
		pos_dict = dict()
		with open(POS) as file:
			for line in file:
				line = line.strip().split()
				pos_dict[line[0]] = int(line[1])
		
		return pos_dict

	def vocabsLabels(self):
		
		labels_dict = dict()
		with open(LABELS) as file:
			for line in file:
				line = line.strip().split()
				labels_dict[line[0]] = int(line[1])
		
		return labels_dict

	def vocabsActions(self):
		
		actions_dict = dict()
		with open(ACTIONS) as file:
			for line in file:
				line = line.strip().split()
				actions_dict[line[0]] = int(line[1])
		
		return actions_dict

	def trainSet(self):

		word_dict = self.vocabsWord
		pos_dict = self.vocabsPos
		labels_dict = self.vocabsLabels
		actions_dict = self.vocabsActions
		train_set = []
		with open(TRAIN) as file:
			data = file.readlines()
			for sample in data:
				feature_list = sample.strip().split()
				for feature in feature_list:
					index = feature_list.index(feature)
					if index < 20:
						word = feature_list[index]
						if word in word_dict.keys():
							feature_list[index] = word_dict[feature]
						else:
							feature_list[index] = word_dict['<unk>']
					elif index < 40:
						feature_list[index] = pos_dict[feature]
					elif index < 52:
						feature_list[index] = labels_dict[feature]
					else:
						feature_list[index] = actions_dict[feature]
				train_set.append(feature_list)

		# data generation
		train_set = np.matrix(train_set) # 143758, 53
		train_size, x_size = train_set.shape # 143758, 53
		all_x = train_set[:,:x_size-1].tolist() # 143758, 52
		all_y = train_set[:,x_size-1:x_size] # 143758, 1
		# one hot encoding
		actions_dim = len(self.vocabsActions.keys()) # 93
		all_y = np.eye(actions_dim)[all_y] # 143758, 1, 93
		all_y = np.reshape(all_y, (train_size, actions_dim)).tolist() # 143758, 93

		return all_x, all_y
	
	def devSet(self):

		word_dict = self.vocabsWord
		pos_dict = self.vocabsPos
		labels_dict = self.vocabsLabels
		actions_dict = self.vocabsActions
		dev_set = []
		with open(DEV) as file:
			data = file.readlines()
			for sample in data:
				feature_list = sample.strip().split()
				for feature in feature_list:
					index = feature_list.index(feature)
					if index < 20:
						word = feature_list[index]
						if word in word_dict.keys():
							feature_list[index] = word_dict[feature]
						else:
							feature_list[index] = word_dict['<unk>']
					elif index < 40:
						pos = feature_list[index]
						if pos in pos_dict.keys():
							feature_list[index] = pos_dict[feature]
						else:
							feature_list[index] = pos_dict['<null>']
					elif index < 52:
						labels = feature_list[index]
						if labels in labels_dict.keys():
							feature_list[index] = labels_dict[feature]
						else:
							feature_list[index] = labels_dict['<null>']
					else:
						actions = feature_list[index]
						if actions in actions_dict.keys():
							feature_list[index] = actions_dict[feature]
						else:
							feature_list[index] = 'NA'
				if feature_list[-1] == 'NA':
					pass
				else:
					dev_set.append(feature_list)

		# data preprocessing
		dev_set = np.matrix(dev_set) # 80065, 53
		dev_size, x_size = dev_set.shape # 80065, 53
		all_x = dev_set[:,:x_size-1].tolist() # 80065, 52
		all_y = dev_set[:,x_size-1:x_size] # 80065, 1
		# one hot encoding
		actions_dim = len(self.vocabsActions.keys()) # 93
		all_y = np.eye(actions_dim)[all_y] # 80065, 1, 93
		all_y = np.reshape(all_y, (dev_size, actions_dim)).tolist() # 80065, 93


		return all_x, all_y

	def embeddingLayer(self, word_inputs, pos_inputs, labels_inputs, word_embed_dim = 64, pos_embed_dim = 32, labels_embed_dim = 32):

		train_size = tf.shape(word_inputs)[0] # batch size
		_, word_feat_dim = word_inputs.get_shape() # _, 20
		_, pos_feat_dim = pos_inputs.get_shape() # _, 20
		_, labels_feat_dim = labels_inputs.get_shape() # _, 12

		word_dim = len(self.vocabsWord.keys()) # 4807
		pos_dim = len(self.vocabsPos.keys()) # 45
		labels_dim = len(self.vocabsLabels.keys()) # 46

		with tf.name_scope('embedingLayer'):
			word_embeding = tf.get_variable('word_embeding', shape=[word_dim, word_embed_dim]) # 4807, 64
			pos_embeding = tf.get_variable('pos_embeding', shape=[pos_dim, pos_embed_dim]) # 45, 32
			labels_embeding = tf.get_variable('labels_embeding', shape=[labels_dim, labels_embed_dim]) # 46, 32

			word_embed = tf.nn.embedding_lookup(word_embeding, word_inputs, name='word_embed') # _, 20, 64
			pos_embed = tf.nn.embedding_lookup(pos_embeding, pos_inputs, name='pos_embed') # _, 20, 32
			labels_embed = tf.nn.embedding_lookup(labels_embeding, labels_inputs, name='labels_embed') # _, 12, 32

			word_embed = tf.reshape(word_embed, [train_size, word_feat_dim*word_embed_dim]) # _, 1280
			pos_embed = tf.reshape(pos_embed, [train_size, pos_feat_dim*pos_embed_dim]) # _, 640
			labels_embed = tf.reshape(labels_embed, [train_size, labels_feat_dim*labels_embed_dim]) # _, 384
			
			embedding_output = tf.concat([word_embed, pos_embed, labels_embed], 1) # _, 2304

		return embedding_output # _, 2304

	def hiddenLayer_1(self, x, hid_dim = 400):

		_, input_dim = x.get_shape() # _, 2304
		
		with tf.variable_scope('hiddenLayer_1'):
			w = tf.get_variable("weights", shape=[input_dim, hid_dim]) # 2304, 200
			b = tf.get_variable("bias", shape=[hid_dim]) # 200

		return tf.nn.relu(tf.matmul(x, w) + b) # _, 200

	def hiddenLayer_2(self, x, hid_dim = 400):

		_, input_dim = x.get_shape() # _, 200

		with tf.variable_scope('hiddenLayer_2'):
			w = tf.get_variable("weights", shape=[input_dim, hid_dim]) # 200, 200
			b = tf.get_variable("bias", shape=[hid_dim]) # 200

		return tf.nn.relu(tf.matmul(x, w) + b) # _, 200

	def outputLayer(self, x):

		_, input_dim = x.get_shape() # _, 200
		actions_dim = len(self.vocabsActions.keys()) # 93
		
		with tf.variable_scope('outputLayer'):
			w = tf.get_variable("weights", shape=[input_dim, actions_dim]) # 200, 93
			b = tf.get_variable("bias", shape=[actions_dim]) # 93

		return tf.nn.softmax(tf.matmul(x, w) + b) # _, 93
	
	def modelGraph(self):

		# define place holder
		train_set, _= self.trainSet
		train_size = len(train_set)
		x_size = len(train_set[0])
		actions_dim = len(self.vocabsActions.keys())
		x = tf.placeholder(tf.int32, shape=[None, x_size]) # _, 52
		y_ = tf.placeholder(tf.float32, shape=[None, actions_dim]) # _, 93
		train_word_inputs = x[:,:20] # _, 20
		train_pos_inputs = x[:,20:40] # _, 20
		train_labels_inputs = x[:,40:52] # _, 12
		# id2vec embedding layer
		embedding_output = self.embeddingLayer(train_word_inputs, train_pos_inputs, train_labels_inputs) # _, 2304
		# first hidden layer
		hidden_layer_1_output = self.hiddenLayer_1(embedding_output) # _, 200
		# second hidden layer
		hidden_layer_2_output = self.hiddenLayer_2(hidden_layer_1_output) # _, 200
		# output layer softmax 
		y = self.outputLayer(hidden_layer_2_output) # _, 93

		return x, y, y_

	def trainer(self, minibatch_size=1000):

		# train set generation & preprocessing
		train_x, train_y = self.trainSet
		train_size = len(train_x)
		# dev set generation & preprocessing
		dev_x, dev_y = self.devSet
		# load model
		x, y, y_ = self.modelGraph
		# backward propagation
		loss = -tf.reduce_sum(y_ * tf.log(y))
		train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
		# save variables
		saver = tf.train.Saver()
		# session run
		with tf.Session() as sess:
			if os.path.isfile(CHECK):
				pass
			else:
				sess.run(tf.global_variables_initializer())
				for epoch in range(7):
					train_index = np.arange(train_size)
					while len(train_index) != 0:
						# show progress
						print('Epoch = %d, Training %.2f%%' % (epoch+1, 100. * (1 - len(train_index)/train_size)))
						# random batch sampling
						if len(train_index) < minibatch_size:
							train_random_range = np.random.choice(train_index, len(train_index), replace=False)
						else:
							train_random_range = np.random.choice(train_index, minibatch_size, replace=False)
						train_batch_xs = [train_x[i] for i in train_random_range]
						train_batch_ys = [train_y[i] for i in train_random_range]
						sess.run(train_op, feed_dict={x: train_batch_xs, y_: train_batch_ys})
						train_index = [i for i in train_index if i not in train_random_range]
					# evaluation
					correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
					accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
					train_accuracy, loss_value = sess.run([accuracy, loss], feed_dict = {x: dev_x, y_: dev_y})
					print("Epoch = %d, train accuracy = %.2f%%, loss = %.2f" % (epoch + 1, 100. * train_accuracy, loss_value))
				# save model
				save_path = saver.save(sess, MODEL)
				print("Model saved in path: %s" % save_path)

	def modelTrained(self):

		# restore variables
		saver = tf.train.Saver()
		saver.restore(self.sess, MODEL)

	def decode(self, sample):

		s = time.time()
		self.i += 1
		# load vocabs dict
		word_dict = self.vocabsWord
		pos_dict = self.vocabsPos
		labels_dict = self.vocabsLabels

		for feature in sample:
			index = sample.index(feature)
			if index < 20:
				word = sample[index]
				if word in word_dict.keys():
					sample[index] = word_dict[feature]
				else:
					sample[index] = word_dict['<unk>']
			elif index < 40:
				pos = sample[index]
				if pos in pos_dict.keys():
					sample[index] = pos_dict[feature]
				else:
					sample[index] = pos_dict['<null>']
			else:
				labels = sample[index]
				if labels in labels_dict.keys():
					sample[index] = labels_dict[feature]
				else:
					sample[index] = labels_dict['<null>']

		sample = np.reshape(sample, (1, len(sample)))
		# load model
		x, y, _ = self.modelGraph
		# decode
		best_act = self.sess.run(tf.argmax(y, 1), feed_dict = {x: sample})[0]
		score = [0]*len(self.actions)
		score[best_act] = 1

		print('Decoding... Now at %d Time: %0.4f second' %(self.i, time.time() - s))

		return score

	def score(self, str_features):
		'''
		:param str_features: String features
		20 first: words, next 20: pos, next 12: dependency labels.
		DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
		:return: list of scores
		'''
		# change this part of the code.
		return self.decode(str_features)

def main():

	input_p = os.path.abspath(sys.argv[1])
	output_p = os.path.abspath(sys.argv[2])
	m = depModel()
	# m.trainer()
	Decoder(m.score, m.actions).parse(input_p, output_p)

if __name__ == '__main__':
	main()