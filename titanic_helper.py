#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:49:45 2017

@author: Cláudio Cavalcante
"""
import pandas as pd
import numpy  as np

def prepareTitanicDf( titanic_dataframe ):
    """
    This function prepares the titanic dataframe.
    
    @type pandas.Dataframe
    @param titanic_dataframe
    
    @rtype pandas.Dataframe
    @return titanic_prepared
    """
    sex      = pd.Series( np.where( titanic_dataframe.Sex == "male", 1, 0 ), name = 'Sex' )
    embarked = pd.get_dummies( titanic_dataframe.Embarked, prefix = "Embarked" )
    pClass   = pd.get_dummies( titanic_dataframe.Pclass, prefix = "Pclass" )
    title    = fixTitle(titanic_dataframe) 
    cabin    = fixCabin(titanic_dataframe)
    ticket   = fixTicket( titanic_dataframe )
    imputed  = fixAgeAndFare( titanic_dataframe )
    family   = fixFamilySize( titanic_dataframe )
    
    return pd.concat( 
            [ sex, embarked, pClass, title, cabin, ticket, imputed, family ], 
            axis=1 
        );
    
def fixFamilySize( titanic_dataframe ): 
    family = pd.DataFrame()

    # introducing a new feature : the size of families (including the passenger)
    family[ 'FamilySize' ] = titanic_dataframe[ 'Parch' ] + titanic_dataframe[ 'SibSp' ] + 1
    
    # introducing other features based on the family size
    family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
    family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
    family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
    
    return family;

def fixAgeAndFare( titanic_dataframe ):
    imputed = pd.DataFrame()
    imputed['Age'] = titanic_dataframe.Age.fillna( titanic_dataframe.Age.mean() )
    imputed['Fare'] = titanic_dataframe.Fare.fillna( titanic_dataframe.Fare.mean() )
    

def cleanTicket(ticket):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'
    

def fixTicket( titanic_dataframe ):    
    
    ticket = pd.DataFrame()

    # Extracting dummy variables from tickets:
    ticket[ 'Ticket' ] = titanic_dataframe[ 'Ticket' ].map( cleanTicket )
    return pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' );
    

def fixCabin( titanic_dataframe ):
    """
    Fix the Cabin column from titanic_datafraem and returns it
     
    @type pandas.Dataframe
    @param titanic_dataframe
    
    @rtype pandas.Dataframe
    @return cabin
    """
    cabin = pd.DataFrame()

    # replacing missing cabins with U (for Uknown)
    cabin[ 'Cabin' ] = titanic_dataframe.Cabin.fillna( 'U' )
    
    # mapping each Cabin value with the cabin letter
    cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
    
    # dummy encoding ...
    return pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' );



def fixTitle( titanic_dataframe ):
    """
    Fix the title from 'Name' column and returns it
     
    @type pandas.Dataframe
    @param titanic_dataframe
    
    @rtype pandas.Dataframe
    @return title
    """
    
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
    # Initialize title dataframe
    title = pd.DataFrame()
    
    # Splits the 'name' column
    title[ 'Title' ] = titanic_dataframe[ 'Name' ].map( 
            lambda name: name.split( ',' )[1].split( '.' )[0].strip() 
            )
    title[ 'Title' ] = title.Title.map( Title_Dictionary )
    return pd.get_dummies( title.Title );
