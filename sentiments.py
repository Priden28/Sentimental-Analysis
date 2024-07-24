import numpy as np
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import pickle

# Constants
FILE_NAME = 'Books'
INPUT_FILE = 'Books_small.json'
OUTPUT_FILE = f'./{FILE_NAME}_small.json'

# Define Sentiment class
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

# Define Review class
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:  # Score of 4 or 5
            return Sentiment.POSITIVE

# Define ReviewContainer class
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

# Reading and Filtering Data
data = []
with open(INPUT_FILE, 'r') as f:
    for line in f:
        review = json.loads(line)
        year = int(review['reviewTime'].split(' ')[-1])
        if year == 2014:
            data.append(review)

# Sampling 1000 reviews from 2014
final_data = random.sample(data, 1000)

# Output the length and first 5 reviews
print(len(final_data))
print(final_data[:5])

# Writing the sampled reviews to a new file
with open(OUTPUT_FILE, 'w') as f:
    for review in final_data:
        f.write(json.dumps(review) + '\n')

# Loading reviews from the file and creating Review objects
reviews = []
with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

# Example of accessing a review score
print(reviews[32].score)

# Splitting data into training and test sets
training, test = train_test_split(reviews, test_size=0.33, random_state=42)

# Creating ReviewContainer objects for training and test sets
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

# Evenly distributing reviews
train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

print(train_x[0])
print(train_x_vectors[0].toarray())

# Training SVM classifier
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

# Predicting sentiment for the first test review
print(test_x[0])
print(clf_svm.predict(test_x_vectors[0]))

# Evaluating the classifier's performance
print(clf_svm.score(test_x_vectors, test_y))
print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))

# Testing the classifier on new data
test_set = ['very fun', "bad book do not buy", 'horrible waste of time', 'will never recommend']
new_test_vectors = vectorizer.transform(test_set)
print(clf_svm.predict(new_test_vectors))

# Hyperparameter tuning using GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 4, 8, 16, 32)}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)
print(clf.score(test_x_vectors, test_y))

# Saving the model using pickle
with open('./sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Loading the model using pickle
with open('./sentiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)
