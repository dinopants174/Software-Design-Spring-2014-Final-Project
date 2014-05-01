import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing

import bw_componentrecognition

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def main():
    NUM_TRAIN = bw_componentrecognition.NUM_TRAIN
    N_BINS = 23
    N_HU_MOMENTS = 7
    N_FEATURES = N_BINS + N_HU_MOMENTS

    X, y = bw_componentrecognition.Data.loadTrain(NUM_TRAIN, N_BINS)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    
    clfs = [
        RandomForestClassifier(n_estimators=20),
        ]
    
    param_dists = [
        {"max_depth": [10, 5, 3, None],
          "max_features": sp_randint(1, 11),
          "min_samples_split": sp_randint(1, 11),
          "min_samples_leaf": sp_randint(1, 11),
          "bootstrap": [True, False],
          "criterion": ["gini", "entropy"]},]
        
    
    for clf, param_dist in zip(clfs, param_dists):
        # run randomized search
        n_iter_search = 25
        
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search)

        random_search.fit(X, y)

        report(random_search.grid_scores_)

if __name__ == '__main__':
    main()
