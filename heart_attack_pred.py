import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# building a neural network
def make_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=13))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= "optimizer",loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier

# import data
data = pd.read_csv("Data/heart.csv")

data.head()

# No need to make dummies/convert string to int


X = data.drop(['output'],axis=1).values
y = data['output'].values


# split into test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# hyperparam tuning
classifier = KerasClassifier(build_fn = make_classifier)

params = {
    'batch_size':[20,35],
    'epochs':[100,200,300,400],
    'optimizer':['adam','rmsprop']
}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring="accuracy",
                           cv=2)


grid_search = grid_search.fit(X_train,y_train)


best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_

best_accuracy
best_param



#found best one, now fit
classifier_best_fit = KerasClassifier(build_fn = make_classifier,batch_size = 25, epochs = 100, optimizer = 'rmsprop')
classifier_best_fit.fit(X_train, y_train)
y_pred = classifier_best_fit.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
cm

accuracies = cross_val_score(estimator = classifier_best_fit,X = X_train,y = y_train,cv = 10,n_jobs = -1)

mean = accuracies.mean()
mean