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

################
# helper classes
################

import plot_functions
import dataframe_functions

resources_folder = "resources/"

train = pd.read_csv( resources_folder + "train.csv" )

################
# Data Preparation
################

"""
These preparations were seen from a kaggle's kernel

"""

