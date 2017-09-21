# Sentiment Analysis
A stochastic gradient descent classifier for sentiment analysis based on four methods of data representation.
## Dependencies
+ OS
+ String
+ NumPy
+ Pandas
+ NLTK
+ Scikit-learn
## Data Representation
+ Unigram
+ Unigram with inverse term frequency
+ Bigram
+ Bigram with inverse term frequency
## Usage Instructions
Simply run:
```
python train.py
```
+ Data preprocessing
+ Data loading
+ Training
## Results
The results will be restored as:
+ output/unigram.output.txt
+ output/unigramtfidf.output.txt
+ output/bigram.output.txt
+ output/bigramtfidf.output.txt

## Score:
+ The score based on unigram: 84.976
+ The score based on unigram_tfidf: 87.26
+ The score based on bigram: 85.26
+ The score based on bigram_tfidf: 85.948
