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
sample_sub = pd.read_csv('SampleSubmissionStage1.csv')

# subset columns and only use 2013+ data
tourney = tourney[tourney.Season > 2013]
tourney = tourney[['Season', 'WTeamID', 'LTeamID']]

# subset ranking day and only use 2013+ data
rankings = rankings[rankings.RankingDayNum == 133]
rankings = rankings[rankings.Season > 2013]

# array of ranking methods and years
methods = rankings.SystemName.unique()
years = tourney.Season.unique()

# initialize empty results dataframe
results = pd.DataFrame()

# loop through each method and year
for method in methods:
    for year in years:
        
        # subset rankings by method and year
        rank = rankings[rankings.SystemName == method]
        rank = rank[rank.Season == year]

        # create dictionary of rank for each team
        rank_dict = pd.Series(rank.OrdinalRank.values, index=rank.TeamID).to_dict()

        # subset tourney matches by year
        match = tourney[tourney.Season == year]

        # map each team's rank
        match['W_rank'] = match['WTeamID'].map(rank_dict)
        match['L_rank'] = match['LTeamID'].map(rank_dict)
        
        # calculate % of matches correctly predicted by ranks
        result = len(match[match.W_rank < match.L_rank])/len(match)
    
        # append results
        temp = pd.Series([method, year, result], index=['method', 'year', 'result'])
        results = results.append(temp, ignore_index=True)

# change year to an int type
results.year = results.year.astype('int64')

# we assume a result of 0 indicates no rankings for that year, so drop those
results = results[results.result != 0]

# group by and average results to determine top ranking methods
avg = results.groupby(['method']).mean()

# top ranking methods are:
    # DC
    # DOK
    # RT
    # BWE
    # LMC
    # SFX
    # STF
    # TPR
    # ACU
    # LOG

