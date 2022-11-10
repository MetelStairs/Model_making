import pandas as pd
import numpy as np

# load data
data = pd.read_csv("Data/ROCK_OR_MINE.csv")

data.head()

# 0 NA, data looks clean
data.isna().sum().sum()

# encodes the R col to int
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data['R'] = le.fit_transform(data['R'])

# Rock(R) = 1
# Mine(M) = 0

# splits x and y
y = data['R']
x = data.drop('R', axis =1)

# split into test train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=30, min_samples_split=8,min_samples_leaf = 1)
clf = clf.fit(X_train,y_train)

# get accuracy

preds = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, preds)


tree.plot_tree(clf)