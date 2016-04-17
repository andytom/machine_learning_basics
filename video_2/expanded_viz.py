import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Test it works
for expected, result in zip(test_target, clf.predict(test_data)):
    print("{} - Expected {} got {}".format(
        expected == result,
        iris.target_names[expected],
        iris.target_names[result]
    ))
