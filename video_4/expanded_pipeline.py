#!/usr/bin/env python


from sklearn import datasets
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# import a dataset
iris = datasets.load_iris()


X = iris.data
y = iris.target


# Set up test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


# Train and test the classifiers
classifiers = {
    "Decision Tree": tree.DecisionTreeClassifier(),
    "KNeighbors": KNeighborsClassifier()
}


results = {}


for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    results[name] = accuracy_score(y_test, predictions)


# Sort and print the results
sorted_results = sorted(results.items(), key=lambda x:x[1], reverse=True)


for name, results in sorted_results:
    print("{0} scored {1:.3%}".format(name, results))
