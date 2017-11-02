#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:53:01 2017

@author: ais
"""

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

import plot_function

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV



    
########################
## DATA INITIALIZATION
########################


# get titanic & test csv files as a DataFrame
train = pd.read_csv("resources/train.csv")
test  = pd.read_csv("resources/test.csv")

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]

del train , test

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)


########################
## TRUE ACTION!
########################

###
# Unncomment this to plot the vars

# plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )

# plot_categories( titanic , cat = 'Sex' , target = 'Survived' )

# plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )


#####
# Data Preparation

sex = pd.Series( np.where( full.Sex == "male", 1, 0 ), name = "Sex" )

embarked = pd.get_dummies( full.Embarked, prefix = "Embarked" )

pClass = pd.get_dummies( full.Pclass, prefix = "Pclass" )

# >>>>>> Fill missing values <<<<<<<< #

# Create dataset
imputed = pd.DataFrame()

imputed['Age'] = full.Age.fillna( full.Age.mean() )
imputed['Fare'] = full.Fare.fillna( full.Fare.mean() )


# >>>>>> Feature Engineering <<<<<<<< #

title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )

cabin = pd.DataFrame()

# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )

# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )

# dummy encoding ...
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

surName = full.Name.map( lambda name: name.split(" ")[0].replace(",",""))
surName = pd.get_dummies( surName, prefix= "familyNames" )

full_X = pd.concat( [ surName, cabin, ticket,imputed , embarked , sex, family, title] , axis=1 )




# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

from sklearn.neural_network import MLPClassifier
import learning_curve

# model = SVC()
# 0.842696629213 0.720149253731

#model = RandomForestClassifier(n_estimators=100)
#0.985553772071 0.798507462687

# model = GradientBoostingClassifier()
# 0.900481540931 0.772388059701

# model = KNeighborsClassifier(n_neighbors = 3)
# 0.812199036918 0.682835820896

model = LogisticRegression()
# 0.781701444623 0.805970149254

model.fit( train_X, train_y )
# Score the model
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))

learning_curve.plot_learning_curve(model, "Testing Learn curve", train_valid_X, train_valid_y , ylim=(0.7, 1.01), n_jobs=4)

#test_Y = model.predict( test_X )
#passenger_id = full[891:].PassengerId
#test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
#test.shape
#test.head()
#test.to_csv( 'resources/titanic_pred.csv' , index = False )