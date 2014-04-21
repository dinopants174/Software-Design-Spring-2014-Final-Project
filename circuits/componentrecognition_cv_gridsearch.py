# -*- coding: utf-8 -*-
"""
Learns a model for classifying images of resistor vs capcitor symbols.
Adapted from code by pruvolo from learn_digist_cv_gridsearch.py 

@author: rlouie
"""

from componentrecognition import *


from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = loadTrain(NUM_TRAIN, nbins=3)
n_trials = 10
train_percentage = 90
# Set the parameters by cross-validation
tuned_parameters = [{'C': [10**-4, 10**-3, 10**-2, 10**-1, 10**0]}]
test_accuracies = np.zeros(n_trials)

for n in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percentage/100.0, random_state=n)
    model = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=5)
    model.fit(X_train, y_train)
    print model.best_estimator_
    test_accuracies[n] = model.score(X_test, y_test)

print 'Average accuracy is %f' % (test_accuracies.mean())