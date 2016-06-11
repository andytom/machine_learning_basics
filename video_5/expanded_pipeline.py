#!/usr/bin/env python3

from scipy.spatial import distance
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# Define our class
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = float("inf")

        for i, train in enumerate(self.X_train):
            dist = distance.euclidean(row, train)
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]

# import a dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Set up test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Setup and train our classifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

# Predict some stuff
predictions = my_classifier.predict(X_test)

score = accuracy_score(y_test, predictions)
print("ScrappyKNN accuracy score is {:.3%}".format(score))
