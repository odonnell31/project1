# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:03:24 2019
@author: Michael O'Donnell
"""

# Description
### This system recommends all-time classic books to readers

### import needed libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
 
### load data in from csv to pandas dataframe
book_data = pd.read_csv("book_data.csv")

### create a user-item matrix from the dataframe
book_matrix = book_data.pivot_table(index='Reviewer', columns='Book',
                                    values='Score')

### break ratings into training and testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(book_matrix, test_size=0.2)

### calculate the raw average of training data
total_sum = (train['Hamlet'].sum(skipna=True) +
      train['Count of Monte Cristo'].sum(skipna=True) +
      train['The Aneid'].sum(skipna=True) +
      train['The Odyssey'].sum(skipna=True) +
      train['The Great Gatsby'].sum(skipna=True))

total_entries = (~np.isnan(train)).sum(1).sum()

train_mean = total_sum/total_entries

print "training raw average: ", train_mean

### calculating rmse for training data
stan_err = []
for i in train.columns[1:5]:
    for j in train[i]:
       if  np.isnan(j) == False: 
           stan_err.append(j-train_mean)
           
train_rmse = sum(stan_err)/len(stan_err)

print "training RMSE: ", train_rmse

stan_err = []
for i in test.columns[1:5]:
    for j in test[i]:
       if  np.isnan(j) == False: 
           stan_err.append(j-train_mean)
           
test_rmse = sum(stan_err)/len(stan_err)

print "testing RMSE: ", test_rmse

# calculating the bias for each user

for user in train.columns[1:5]:
    print user, train[user].mean(skipna=True)
    
# calculating the bias for each item

print train.mean(axis=1, skipna=True)