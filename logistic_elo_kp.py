#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 20:59:38 2018

@author: jordanbaker
"""

# load base packages
import os
import numpy as np
import pandas as pd

# change path
path = '/Users/jordanbaker/Documents/Data Science/kaggle_march_madness/DataFiles'
os.chdir(path)

# load data
kp = pd.read_csv('ken_pom.csv')
elo = pd.read_csv('season_elos.csv')
tourney = pd.read_csv('NCAATourneyCompactResults.csv')
sample_sub = pd.read_csv('SampleSubmissionStage1.csv')
gt = pd.read_csv('gtsubmission.csv')

# rename columns for merging
gt.rename(columns={'Pred': 'outcome'}, inplace=True)
kp.rename(columns={'year': 'season', 'teamID': 'team_id'}, inplace=True)

# merge elo stats and ken pomeroy ratings
both = pd.merge(kp, elo,  how='left', on=['team_id','season'])
both = both[both.season > 2012]

# extract year, Team A, and Team B from ID of sample submission file
sample_sub['Year'] = sample_sub['ID'].str.split('_').str[0]
sample_sub['Team_A'] = sample_sub['ID'].str.split('_').str[1]
sample_sub['Team_B'] = sample_sub['ID'].str.split('_').str[2]

# convert extracted features to ints
sample_sub.Year = sample_sub.Year.astype('int64')
sample_sub.Team_A = sample_sub.Team_A.astype('int64')
sample_sub.Team_B = sample_sub.Team_B.astype('int64')

# list of years
years = both.season.unique()

# initialize empty df
df = pd.DataFrame()

# loop through by year to pull elo stats and kp ratings for all years
for year in years:
    
    # subset rankings by year
    temp = both[both.season == year]

    # subset matches by year
    sub = sample_sub[sample_sub.Year == year]

    # create dictionary of elo and rank for each team
    elo_dict = pd.Series(temp['season_elo'].values, index=temp.team_id).to_dict()
    rank_dict = pd.Series(temp['rank'].values, index=temp.team_id).to_dict()

    # map each team's rating and rank
    sub['A_elo'] = sub['Team_A'].map(elo_dict)
    sub['B_elo'] = sub['Team_B'].map(elo_dict)
    sub['A_rank'] = sub['Team_A'].map(rank_dict)
    sub['B_rank'] = sub['Team_B'].map(rank_dict)

    # append rankings
    df = df.append(sub, ignore_index=True)
    
# merge with ground truth file for modeling
df = gt.merge(df, on='ID', how='left')

# training and testing data
train = df[df.Year == 2016]
test = df[df.Year == 2017]

# setup and train logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(train.iloc[:,-4:], train.iloc[:,1])

# generate predictions
predictions = pd.DataFrame(logreg.predict_proba(test.iloc[:,-4:]))

# filter ground truth file
gt['Year'] = gt['ID'].str.split('_').str[0]
gt.Year = gt.Year.astype('int64')
gt = gt[gt.Year == 2017]
gt = gt.drop('Year', axis=1)

# test logloss of predictions
gt['Pred_y'] = results[1].values
gt.rename(columns={'outcome': 'Pred_x'}, inplace=True)
logloss(gt)


from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression()
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(logreg, param_grid = params, verbose = 5, scoring = "log_loss")
clf.fit(train.iloc[:,-4:], train.iloc[:,1])
results = pd.DataFrame(clf.predict_proba(test.iloc[:,-4:]))
    
    
# GS: 0.4801905230634016
# Reg: 0.4821655510173796
# Reg on multiple years: 0.49720144903976077
    
    