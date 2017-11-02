#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:35:51 2017

These code were implemented by Udacity!!!

@author: Cláudio Cavalcante
"""

from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

def checkPredictions(X_all, y_all, classifiers, data_ranges, train_size ): 
    """
    Iterates over a list of classifiers to check the best performance and 
    prints it the results
    
    @type list
    @param classifiers
    
    @type list
    @param data_ranges
    
    @return void
    """
    X_train, X_test, y_train, y_test = train_test_split(
            X_all,y_all, train_size=train_size
    )
    
    for clf in classifiers:
        for rng in data_ranges:
            
            print( "=================" )
            
            train_predict(clf, X_train[:rng], y_train[:rng], X_test, y_test)
        
    print( "\n\nENDSSS" );

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print( "Trained model in {:.4f} seconds".format(end - start) )

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print( "Made predictions in {:.4f} seconds.".format(end - start) )
    return f1_score(target.values, y_pred, pos_label=1)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print( "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)) )
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print( "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print( "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))