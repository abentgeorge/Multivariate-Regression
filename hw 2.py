# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:54:24 2021

@author: Aben George
"""

# PREDICT SALARY FOR 
# 2y, 9 test score,  6 iv
# 12 yr, 10 test, 10 iv


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# WORD TO NUMBER PYTHON PACKAGE

from word2number import w2n

# Ex - print(w2n.word_to_num('two point three'))

df = pd.read_csv('hiring.csv')
df['experience'][2] = w2n.word_to_num(df['experience'][2])

df['experience'][3], df['experience'][4] = w2n.word_to_num(df['experience'][3]), w2n.word_to_num(df['experience'][4])

df['experience'][5], df['experience'][6], df['experience'][7] = w2n.word_to_num(df['experience'][5]), w2n.word_to_num(df['experience'][6]), w2n.word_to_num(df['experience'][7])

mean_bed = df.experience.mean()

# -------------------------------------------------------------

import warnings


import pandas as pd
pd.options.mode.chained_assignment = None
# -------------------------------------------------------------

df['experience'] = df['experience'].fillna(0)

import math
meantest= math.floor(df['test_score(out of 10)'].mean())

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(7)
#------------------------------------------------------------------------


reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],df['salary($)'])

print(reg.predict(np.array([2,9,6]).reshape(1,-1)))

print(reg.predict(np.array([12,10,10]).reshape(1,-1)))
