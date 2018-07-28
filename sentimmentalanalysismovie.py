
"""
Created on Sat Jul 28 14:45:12 2018

@author: Preetham

"""

import sklearn
from sklearn.datasets import load_files

moviedir = r'D:\movie_reviews'
movie_train = load_files(moviedir, shuffle=True)                                # loading all files as training data. 
len(movie_train.data)
movie_train.target_names                                                        # target names ("classes") are automatically generated from subfolder names
movie_train.data[0][:500]                                                       #first file i.e about a movie

movie_train.filenames[0]                                                        # first file is in "neg" folder
movie_train.target[0]                                                           # first file is a negative review and is mapped to 0 index 'neg' in target_names

from sklearn.feature_extraction.text import CountVectorizer
import nltk
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)            # initialize movie_vector object, and then turn movie train data into a vector
movie_counts = movie_vec.fit_transform(movie_train.data)
movie_vec.vocabulary_.get('screen')                                            # 'screen' is found in the corpus, mapped to index 19637



movie_vec.vocabulary_.get('seagal')                                            # similarly Mr. Steven Seagal is present
movie_counts.shape                                                             #2000 documents,25k words

# Convert raw frequency counts into TF-IDF values

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

movie_tfidf.shape                                                              # Same dimensions, now with tf-idf values instead of raw frequency counts

#training and testing using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB                                  #using Multinominal Naive Bayes as our model

from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0.20, random_state = 12)      #Split data into training and test sets

clf = MultinomialNB().fit(docs_train, y_train)                                 #training data

y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)                                 # Predicting the Test set results, find accuracy



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# very short and fake movie reviews
reviews_new = ['I hate this movie god!']                                        #input your review here
reviews_new_counts = movie_vec.transform(reviews_new)
reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts) 

pred = clf.predict(reviews_new_tfidf)                                          #predict negetive or positive

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie_train.target_names[category]))