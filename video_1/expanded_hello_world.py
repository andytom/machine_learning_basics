from sklearn import tree

# Name Lookup
NAMES = {
    0: 'Apple',
    1: 'Orange'
}

# Training Data
features = [
    [140, 1],
    [130, 1],
    [150, 0],
    [170, 0]
]

labels = [
    0,
    0,
    1,
    1
]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Classify some stuff
tests = [
    [160, 0],
    [120, 1],
    [145, 1],
    [145, 0],
    [160, 1],
    [120, 0],
]

results = clf.predict(tests)

for test, res in zip(tests, results):
    name = NAMES[res]
    print("{} is an {}".format(test, name))
