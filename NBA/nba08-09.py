# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:01:29 2026

@author: Micah
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
os.chdir("C:\\Users\\Micah\\Desktop\\PythonBA")


df = pd.read_csv('nbaShots08_09.csv')

# Insight 1
# Do 2 pointers go in more often than 3 pointers?

shot_type_summary = df.groupby("SHOT_TYPE")["SHOT_MADE_FLAG"].mean() * 100
shot_type_summary = shot_type_summary.round(1)

print(shot_type_summary)

# 2 pointers have a 48.5% make rate while 3 pointers have a 36.7% make rate

# Insight 2
# As shot distance increases, does the make rate go down?

X = df[["SHOT_DISTANCE"]]
y = df[["SHOT_MADE_FLAG"]]

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# At 0 feet, you have a predicted make rate of 56.6%
# Each extra foot drops that by -0.9%
