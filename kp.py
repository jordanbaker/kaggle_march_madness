#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:00:39 2018

@author: jordanbaker
"""

# load base packages
import os
import numpy as np
import pandas as pd

# change path
path = "/Users/jordanbaker/Documents/Data Science/kaggle_march_madness/DataFiles"
os.chdir(path)

# load data
kp = pd.read_csv('ken_pom.csv')
sample_sub = pd.read_csv('SampleSubmissionStage1.csv')

# list of years
years = kp.year.unique()

# extract year, Team A, and Team B from ID of sample submission file
sample_sub['Year'] = sample_sub['ID'].str.split('_').str[0]
sample_sub['Team_A'] = sample_sub['ID'].str.split('_').str[1]
sample_sub['Team_B'] = sample_sub['ID'].str.split('_').str[2]

# convert extracted features to ints
sample_sub.Year = sample_sub.Year.astype('int64')
sample_sub.Team_A = sample_sub.Team_A.astype('int64')
sample_sub.Team_B = sample_sub.Team_B.astype('int64')

# del Pred column (pre-filled 0.5 probs) so we can create our own
del sample_sub['Pred']

# initialize empty df for predictions
pred = pd.DataFrame()

# loop through by year to pull kp ratings for all years
for year in years:
    
    # subset rankings by year
    rank = kp[kp.year == year]

    # subset matches by year
    sub = sample_sub[sample_sub.Year == year]

    # create dictionary of rank for each team
    rank_dict = pd.Series(rank['rank'].values, index=rank.teamID).to_dict()

    # map each team's rank
    sub['A_rank'] = sub['Team_A'].map(rank_dict)
    sub['B_rank'] = sub['Team_B'].map(rank_dict)
        
    # append rankings
    pred = pred.append(sub, ignore_index=True)
    
# function to assign a win (1) to the lower ranked team, by method
def predict_win(data):
    
    if (data['A_rank'] < data['B_rank']):
        return 1
    else:
        return 0

# use previously defined function to predict wins for all matches
pred['Pred'] = pred.apply(predict_win, axis = 1)
    
# alter probabilities to minimize/maximize score
pred.loc[pred.Pred == 0, 'Pred'] = .22
pred.loc[pred.Pred == 1, 'Pred'] = .72

# create final submission file
final = pred[['ID', 'Pred']]
final.to_csv('final.csv')
