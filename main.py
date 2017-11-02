#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:32:26 2017

@author: ais
"""

################
# Imports
################

import numpy as np
import pandas as pd
import gc

################
# helper classes
################

import plot_functions
import dataframe_functions
import titanic_helper
from classifier_helpers import checkPredictions

resources_folder = "resources/"

train = pd.read_csv( resources_folder + "train.csv" )
titanic, survived = dataframe_functions.splitColumn( train, "Survived" )

del train
gc.collect()

fixed = titanic_helper.prepareTitanicDf(titanic)

################
# Classifiers
################

from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
#
clf_A = RandomForestClassifier()
#clf_B = DecisionTreeClassifier()
#clf_C = LogisticRegression()
#clf_D = MLPClassifier()
#
#classifiers = [ clf_A, clf_B, clf_C, clf_D ]
#ranges      = [ 100, 400, 600 ]
#
#
#checkPredictions(fixed, survived,classifiers, ranges, .7)

from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
import scorer

def f1_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return f1


# TODO: Create the parameters list you wish to tune
parameters = {
    "n_estimators":[5, 10, 15, 25, 45],
    "criterion": ["gini", "entropy"],
    "max_depth": [500, 800, 1000, 3000], 
    "min_samples_split":[2,6,16]
}

X_train, X_test, y_train, y_test = train_test_split(
            fixed,survived, train_size=.7, random_state=42
    )

best_params = scorer.showScorer( X_train, y_train, X_test, y_test, clf_A, parameters, f1_metrics )

#Made predictions in 0.0030 seconds.
#Tuned model has a training F1 score of 0.8184.
#Made predictions in 0.0023 seconds.
#Tuned model has a testing F1 score of 0.7714.
