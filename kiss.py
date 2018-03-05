#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 18:46:40 2018

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
rankings = pd.read_csv('MasseyOrdinals.csv')
tourney = pd.read_csv('NCAATourneyCompactResults.csv')

# subset columns and only use 2011+ data
tourney = tourney[tourney.Season > 2010]
tourney = tourney[['Season', 'WTeamID', 'LTeamID']]

rankings = rankings[rankings.RankingDayNum == 133]
rankings = rankings[rankings.Season > 2010]

methods = rankings.SystemName.unique()
years = tourney.Season.unique()


results = pd.DataFrame()

for method in methods:
    for year in years:
        
        rank = rankings[rankings.SystemName == method]
        rank = rank[rank.Season == year]

        rank = pd.Series(rank.OrdinalRank.values, index=rank.TeamID).to_dict()

        match = tourney[tourney.Season == year]

        match['W_rank'] = match['WTeamID'].map(dictionary)
        match['L_rank'] = match['LTeamID'].map(dictionary)
        
        result = len(match[match.W_rank < match.L_rank])/len(match)
    
        temp = pd.Series([method, year, result], index=['method', 'year', 'result'])
        results = results.append(temp, ignore_index=True)

results.year = results.year.astype('int64')
