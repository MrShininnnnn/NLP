# Sequence Learning for Semantic Parsing
The project is for the semantic parsing task. The model aims to parse a natural language sentence to a semantic representation.
## Getting Started
The following instructions will ensure a copy of the project can run well on any local machine for development and testing purposes. 
### Prerequisites
A virtual environment with python 3.6 is recommended. The required environment with necessary libraries include:
* Python >= 3.5
* Tensorflow/ Tensorflow-gpu >= 1.12.0
* Numpy >= 1.14.5
```
$ pip3 install pip --upgrade
$ pip3 install -r requirement.txt
```
### Directory Structure
Here is a graph to show the entire folder structure.
```
Semantic_Parsing/
├── README.md
├── requirements.txt
├── src
│   ├── model
│   │   └── att_bi_lstm_s2s.py
│   ├── source
│   │   ├── config.py
│   │   ├── data
│   │   │   └── job640
│   │   │       ├── pp_data.json
│   │   │       └── vocab.json
│   │   └── log
│   │       └── att_bi_lstm_s2s
│   └── toolkit
│       ├── eva.py
│       └── load.py
└── train.py
```
### Folder List
* src/model - model graphs
* src/source/data - required data
* src/source/log - logs for model saving points
* src/toolkit - tools for loading file and evaluation
### File List
* train.py - to train the model
* config.py - default configuration
## Testing
Under the root directory, simply run:
```
:$ python3 train.py
```
## Result
In this project, the model is tested on a popular benchmark dataset, JOBs, which contains 640 pairs of a sentence and its logic representation. The final result of the model should achieve around 90% sequence accuracy.
## Author
* **Ning Shi** - Shining
* **ning.shi@nyu.edu**
