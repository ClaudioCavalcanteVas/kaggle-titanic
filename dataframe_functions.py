#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:34:51 2017

Some DataFrame helper functions

@author: ais
"""

def splitColumn( dataFrame, targetColumn, axis = 1 ):
    """
    This function splits the dataFrame by a target column and 
    returns a tuple with these values
    
    @type pandas DataFrame
    @param dataFrame    
    
    @type string | list
    @param targetColumn
    
    @rtype tuple
    @return returns a dataFrame split
    
    """
    return [ 
            dataFrame.drop(targetColumn, axis=axis), 
            dataFrame[targetColumn]  
           ]

