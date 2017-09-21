# Large Movie Review Dataset v1.0
[Data download here](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
## Overview
This dataset contains movie reviews along with their associated binary sentiment polarity labels. It is intended to serve as a benchmark for sentiment classification. This document outlines how the dataset was gathered, and how to use the files provided. 
## Dataset
The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). We also include an additional 50,000 unlabeled documents for unsupervised learning.  

In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings. Further, the train and test sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their associated with observed labels.  In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets. In the unsupervised set, reviews of any rating are included and there are an even number of reviews > 5 and <= 5.
## Files
Please download the data and unzip it in this folder. You should get the following directory structure:

  + data/aclImdb/train/neg
  + data/aclImdb/train/pos
  + data/aclImdb/test/pos
  + data/aclImdb/test/pos
  
The sub-directories pos/ is for positive texts and neg/ is for negative one.  
The code will combine the raw database into two single csv files:

  + data/train/imdb_tr.csv
  + data/test/imdb_te.csv
  
The csv file should have three columns:

  + "row_number"
  + "text"
  + "polarity"
  
The column "text" contains review texts from the aclImdb database and the column "polarity" consists of sentiment label, 1 for positive and 0 for negative.
## Reference
Potts, Christopher. 2011. On the negativity of negation. In Nan Li and David Lutz, eds., Proceedings of Semantics and Linguistic Theory 20, 636-659.
## Contact
For questions/comments/corrections please contact Andrew Maas amaas@cs.stanford.edu
