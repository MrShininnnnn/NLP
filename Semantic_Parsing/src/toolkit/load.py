#! /usr/bin/python3.5
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@nyu.edu'

import json

def loadJson(path):

	with open(path, 'r') as f:
		return json.load(f)

def loadText(path):

	with open(path, 'r') as f:
		return f.read().splitlines()
