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
methods = list(rankings.SystemName.unique())
years = tourney.Season.unique()

methods_full = pd.DataFrame()

for method in methods:
    
    temp_df = rankings[rankings.SystemName == method]
    temp_size = len(temp_df.Season.unique())
    
    # append results
    temp = pd.Series([method, temp_size], index=['method', 'size'])
    methods_full = methods_full.append(temp, ignore_index=True)


methods_full = methods_full.loc[methods_full['size'] == 4]

methods = methods_full.method.unique()

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
    # 
    # 
    #
    # 
    # 
    # 
    # 
    # 
    # 
    # 

  
# generate list of top ranking methods
# subset rankings by those top ones
top = ['7OT', 'RTP', 'STH', 'LMC', 'CRO', 'BBT', 'DC', 'KPK', 'SAG', 'BUR'] 
# top = ['7OT']
# top = rankings.SystemName.unique()
#top = list(avg.index)
#top.remove('USA')
#top.remove('AP')
#top.remove('DES')

# subset rankings by those top ones
sub_rankings = rankings[rankings.SystemName.isin(top)]   

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
    rank = sub_rankings[sub_rankings.Season == year]

    # subset tourney matches by year
    sub = sample_sub[sample_sub.Year == year]

    for method in top:
        
        # subset rankings by method
        temp_rank = rank[rank.SystemName == method]

        # create dictionary of rank for each team
        rank_dict = pd.Series(temp_rank.OrdinalRank.values, index=temp_rank.TeamID).to_dict()

        # map each team's rank
        sub[method + '_A_rank'] = sub['Team_A'].map(rank_dict)
        sub[method + '_B_rank'] = sub['Team_B'].map(rank_dict)
        
    pred = pred.append(sub, ignore_index=True)
    
    
def abc(data):
    
    if (data[method + '_A_rank'] < data[method + '_B_rank']):
        return 1
    else:
        return 0
    
        
for method in top:
    pred[method + '_Choice'] = pred.apply(abc,axis = 1)
    
pred['Pred'] = pred.iloc[:,-len(top):].sum(axis=1)/len(top)

pred.loc[pred.Pred == 0, 'Pred'] = .22
pred.loc[pred.Pred == 1, 'Pred'] = .77

# random resting code
# temp = rankings[rankings.Season == 2014]
# temp = temp[temp.SystemName == 'BBT']
# temp = temp[temp.OrdinalRank == 1112]
len(pred[pred.Pred == 0])/len(pred)


final = pred[['ID', 'Pred']]
final.to_csv('final.csv')

