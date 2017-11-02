#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:37:53 2017

@author: ais
"""
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from classifier_helpers import predict_labels
    
def showScorer(X_train, y_train, X_test, y_test, classifier, parameters, metrics ):
    
    # TODO: Make an f1 scoring function using 'make_scorer' 
    scorer = make_scorer(metrics)
    
    # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
    grid_obj = GridSearchCV(classifier, parameters,scoring = scorer)
    
    # TODO: Fit the grid search object to the training data and find the optimal parameters
    grid_obj = grid_obj.fit(X_train, y_train)
    
    # Get the estimator
    classifier = grid_obj.best_estimator_

    
    # Report the final F1 score for training and testing after parameter tuning
    print(
            "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(classifier, X_train, y_train))
        )
    print( "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(classifier, X_test, y_test)))
    return grid_obj.best_params_;