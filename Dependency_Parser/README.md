# Dependency Parser
A dependency parser model based on transitions extracted from trees in each sentence.
## Dependencies
+ python>=3.6.4
+ numpy>=1.14.1 
+ tensorflow>=1.15.4
## Directory Structure
	|root
		|data (train set, dev set, and vocabs set)
		|output (predicted outputs for both dev set and test set)
		|src (most python code)
			|__pycache__ (cache)
			|tmp (saved model parameters)
				|depModel1
				|depModel2
				|depModel3
		|trees (original tree structure data set)
## Vocabulary Creation for Features
```
$ python2 src/gen_vocab.py trees/train.conll data/vocabs
```
## Data Generation
```
$ python2 src/gen.py trees/train.conll data/train.data
$ python2 src/gen.py trees/dev.conll data/dev.data

```
## Usage Instructions
Simply run:
```
$ python3 src/depModel3.py
```
## depModel1
	Parameter:
		Trainer = Adam algorithm
		Learning rate = 1e-4
		Training epochs = 7
		Transfer function = RELU
		Word embedding dimension = 64
		POS embedding dimension = 32
		Dependency embedding dimension = 32
		Mini-batch size = 1000
		First hidden layer dimension = 200
		Second hidden layer dimension = 200
	Training Progress:
		Epoch = 1, train accuracy = 53.84%, loss = 158501.31
		Epoch = 2, train accuracy = 69.30%, loss = 106213.09
		Epoch = 3, train accuracy = 76.41%, loss = 74347.81
		Epoch = 4, train accuracy = 81.60%, loss = 55592.02
		Epoch = 5, train accuracy = 85.42%, loss = 44570.02
		Epoch = 6, train accuracy = 87.06%, loss = 37859.88
		Epoch = 7, train accuracy = 88.49%, loss = 33337.14
	Unlabeled attachment score: 74.93
	Labeled attachment score: 67.88
## depModel2
	Parameter:
		Trainer = Adam algorithm
		Learning rate = 1e-4
		Training epochs = 7
		Transfer function = RELU
		Word embedding dimension = 64
		POS embedding dimension = 32
		Dependency embedding dimension = 32
		Mini-batch size = 1000
		First hidden layer dimension = 400
		Second hidden layer dimension = 400
	Training Progress:
		Epoch = 1, train accuracy = 64.76%, loss = 120673.73
		Epoch = 2, train accuracy = 78.09%, loss = 71082.94
		Epoch = 3, train accuracy = 84.03%, loss = 49332.62
		Epoch = 4, train accuracy = 87.08%, loss = 38532.09
		Epoch = 5, train accuracy = 88.69%, loss = 32418.34
		Epoch = 6, train accuracy = 89.97%, loss = 28567.11
		Epoch = 7, train accuracy = 90.41%, loss = 26230.74
	Unlabeled attachment score 77.49
	Labeled attachment score 72.22
## depModel3
	Parameter:
		Trainer = Adam algorithm
		Learning rate = 1e-4
		Training epochs = 8
		Transfer function = RELU
		Word embedding dimension = 128
		POS embedding dimension = 64
		Dependency embedding dimension = 64
		Mini-batch size = 512
		First hidden layer dimension = 800
		Second hidden layer dimension = 800
	Training Progress:
		Epoch = 1, train accuracy = 84.11%, loss = 48136.61
		Epoch = 2, train accuracy = 89.86%, loss = 28887.05
		Epoch = 3, train accuracy = 91.36%, loss = 23144.54
		Epoch = 4, train accuracy = 91.83%, loss = 21492.03
		Epoch = 5, train accuracy = 92.33%, loss = 20223.92
		Epoch = 6, train accuracy = 92.81%, loss = 18910.97
		Epoch = 7, train accuracy = 92.96%, loss = 18719.42
		Epoch = 8, train accuracy = 93.10%, loss = 18470.00
	Unlabeled attachment score: 83.09
	Labeled attachment score: 79.87
