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

kp = pd.read_csv('ken_pom.csv')
sample_sub = pd.read_csv('SampleSubmissionStage1.csv')

years = kp.year.unique()

sample_sub['Year'] = sample_sub['ID'].str.split('_').str[0]
sample_sub['Team_A'] = sample_sub['ID'].str.split('_').str[1]
sample_sub['Team_B'] = sample_sub['ID'].str.split('_').str[2]

sample_sub.Year = sample_sub.Year.astype('int64')
sample_sub.Team_A = sample_sub.Team_A.astype('int64')
sample_sub.Team_B = sample_sub.Team_B.astype('int64')
del sample_sub['Pred']

pred = pd.DataFrame()

for year in years:
    
    # subset rankings by year
    rank = kp[kp.year == year]

    # subset tourney matches by year
    sub = sample_sub[sample_sub.Year == year]


    # create dictionary of rank for each team
    rank_dict = pd.Series(rank['rank'].values, index=rank.teamID).to_dict()

    # map each team's rank
    sub['A_rank'] = sub['Team_A'].map(rank_dict)
    sub['B_rank'] = sub['Team_B'].map(rank_dict)
        
    pred = pred.append(sub, ignore_index=True)
    
    
def abc(data):
    
    if (data['A_rank'] < data['B_rank']):
        return 1
    else:
        return 0
    
pred['Pred'] = pred.apply(abc,axis = 1)
    
pred.loc[pred.Pred == 0, 'Pred'] = .22
pred.loc[pred.Pred == 1, 'Pred'] = .72

# random resting code
# temp = rankings[rankings.Season == 2014]
# temp = temp[temp.SystemName == 'BBT']
# temp = temp[temp.OrdinalRank == 1112]
len(pred[pred.Pred == 0])/len(pred)


final = pred[['ID', 'Pred']]
final.to_csv('final.csv')
