# Sentiment Prediction for Amazon Kindle Review

### Introduction
User reviews on E-commerce websites like Amazon.com have a large influence on product reputation as they are heavily viewed by prospective buyers before they decide to make purchases. Text mining techniques and algorithms can help uncover customer attitudes and sentiments on products they have purchased and used. Leveraging a large amount of text-based product review to automatically recognize emotional polarity of a text through the natural language processing and translating the potential business value into realistic business benefits are essential to the E-commerce business. In this project, we try to design a web-based product that incorporates different machine learning algorithms and front-end user interface.

> Topics
> * Web Crawling on Amazon Review Dataset as a model training database
> * Natural Language Processing to transform text-based data into numeric-based data
> * Comparison and Evaluation of Multiple Machine Learning Algorithms 
> * Web user interface design 
> * Sentiment prediction and information retrieval with highlighted key word

### Objective
The goal of this web-based NLP app is not only to classify the sentiment polarity of text paragraph, but also explore the differences of various machine learning algorithm on this specific topic. Furthermore, this completed product shows the product pipeline from data collection, data processing, modeling and implementation of the model into front-end user interface.

During Natural Language Processing, techniques including Stop words, lemmatization, POS tagging, stemming, TF-IDF score and word2vec will be utilized for transformation from unstructured data into structured data for further modeling.

Machine learning algorithms include Naive Bayes, Logistic Regression and Long Short-term Memory with Keras backend to Tensorflow. And each model has its' own advantages and limitations on computing time and prediction accuracy.


### Feature Analysis and Engineering
The data set of comment text and rating score used in this project was downloaded from website http://jmcauley.ucsd.edu/data/amazon/. And we only focused on sentiment for Amazon Kindle category only and 12k sample data were used for training and testing. 

After stemming, lemmatization and tokenization to build corpus, we have the frequency of word for each review. We used the bag-of-words approach for text, a good and simple start, which ignores grammar and even word order. The terms ' Amazon Kindle' and 'Kindle Amazon' have the same probability score. And POS tagging involves tagging every word in the document and assigns part of speech - noun, verb, adjective, pronoun, single noun, plural noun, etc. TF-IDF weights also implemented in our project to reflect how important a word is to a product review in the corpus, which helps to adjust for the fact that some words appear more frequently in general. N-grams are basically a set of co-occuring words within a given window. Both uni-gram and bi-gram were tried in our case to build features.


### Modeling
* Multinomial NB

Since our project is a classification problem for word counts with tf-idf weighting, multinomial Naïve Bayes classifier is well suited in this case. Naïve Bayes classifier is a simple classification method based on Bayes rul. It relies on very simple representation of document, e.g. Bag of words. The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

* SVM

Linear SVM is another good option for text classification. Most of text classification problems are linearly separable. And the linear kernel is good when there is a lot of features. That's because mapping the data to a higher dimensional space does not really improve the performance. In text classification, both the numbers of instances (document) and features (words) are large. Furthermore, The Linear Kernel is computationally very cheap and less parameters to optimize. Only C regularization parameter need to be tuned in linear SVM.
Logistic regression
Logistic regression is a Discriminative model to estimates the probability directly from the training data by minimizing error.

* LSTM RNN

Word2vec is a group of neural networks that are trained to produce a vector space for each word in the corpus instead of each document. Wording embedding capture the relationship between words and words.
Recurrent nets are a type of artificial neural network designed to recognize patterns in sequences of data. LSTM is a special kind of RNN. It is explicitly designed to avoid the long-term dependency problem. LSTM is composed of a cell state and gates. Cell state runs straight down the entire chain, with some linear interactions with vectors from gates. Information can be removed or added and the importance of information can also be adjusted via forget gate, input gate and output gate.  

* Results 

Model         | Accuracy
------------- | -------------
Multinomial Naïve Bayes| 0.832
SVM           | 0.83
Logistic regression | 0.83
LSTM RNN      | 0.83


Multinomial Naive Bayes is the winner among the four models with 83% accuracy. SVM, Logistic regression and LSTM are at par on accuracy but SVM is slightly time-consuming than logistic regression, 1.96s vs 1.87s. LSTM has more parameters to tune than other models.


### Reference

[1] A Beginner's Guide to Recurrent Networks and LSTMs. https://deeplearning4j.org/lstm.html

[2] Colah's blog. Understanding LSTM Networks. http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 

