#! /usr/bin/python3.5
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@nyu.edu'

import numpy as np

class Evaluate():

	def __init__(self, targets, predictions):

		self.tars = targets
		self.preds = predictions
		self.size = len(targets)

	def seqCheck(self, tar, pred):

		if len(tar) <= len(pred) and \
			sum(np.equal(np.array(tar), np.array(pred))) == len(tar):
			return True
		else:
			return False

	def acc(self):
		c = 0
		for i in range(self.size):
			tar = self.tars[i]
			pred = self.preds[i]
			if self.seqCheck(tar, pred):
				c += 1
		return np.float32(c / self.size)
