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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

clf_A = RandomForestClassifier()
clf_B = DecisionTreeClassifier()
clf_C = LogisticRegression()
#clf_D = GaussianNB()

classifiers = [ clf_A, clf_B, clf_C ]
ranges      = [ 300, 500, 800 ]


checkPredictions(fixed, survived,classifiers, ranges, .7)


