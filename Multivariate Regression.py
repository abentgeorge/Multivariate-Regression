# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:07:59 2021

@author: Aben George
"""


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#1) DATA PREPROCESSING - REMOVE EMPTY VALUES 

df = pd.read_csv('homeprices.csv')

# DEALING WITH EMPTY VALUE 1 - TAKE MEAN or MEDIAN OF COLUMN | FILLNA WITH VALUE


#print("The mean is : ", df.bedrooms.mean)

# ------------------------------------------------------

#       to remove decimals use math.floor

import math

#mean_bedrooms = math.floor(df['bedrooms'].mean())
median_bedrooms = math.floor(df.bedrooms.median())
mean_bedrooms = math.floor(df.bedrooms.mean())
#print(mean_bedrooms)

mean_bed = df.bedrooms.mean()
#----------------------------------

#       USE .FILLNA()
# NEED TO USE DATAFRAM['BEDROOMS]  AND NOT DIRECT .BEDROOMS


df['bedrooms'] = df['bedrooms'].fillna(3)
#print(df)

# -------------------------------------------------------------

#2) MODELLING

# CREATE REG OBJECT
reg = linear_model.LinearRegression()

# FIT THE DATA - .fit([independant variables], dependant variable)

reg.fit(df[['area', 'bedrooms', 'age']],df['price'])

#print(reg.coef_)

#Q1) PREDICT HOME PRICE OF 3000SQFT, 3 BED, 40 YEAR OLD

print(reg.predict(np.array([3000,3,40]).reshape(1,-1)))

#               remember to change Question values into an array

#Q2) PREDICT HOME PRICE OF 2500SQFT, 4 BED, 5 YEAR OLD

print(reg.predict(np.array([2500,4,5]).reshape(1,-1)))

#------------------------------------------------------------







