#!/usr/bin/env python3
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import yaml
import sys
import os

# The functions
def bar_chart(feature, title):
    result = pd.read_csv('/data/result.csv')
    sns.set_style("dark")
    mask = result.Sex == 0
    result.Sex[mask] = 'Male'
    result.Sex[~mask] = 'Female'
    survived = result[result['Survived']==1][feature].value_counts()
    dead = result[result['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    col = df.columns.to_list()
    for count, value in enumerate(col):
        col[count] = feature + '_' + str(value)
    df.columns = col
    df.plot(kind='bar',stacked=True, figsize=(10,10)).legend(loc='upper center', ncol=len(col), title=f"{feature} type")
    plt.title(f'{title}', fontsize=23)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.xlabel("Survived or Dead", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.savefig(f'/data/{feature}_relation.png')

def pie_chart(feature, title):
    result = pd.read_csv('/data/result.csv')
    sns.set_style("dark")
    survived = result[result['Survived']==1][feature].value_counts()
    dead = result[result['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df = df.fillna(0)
    mask = df.sum() < 0.01*df.values.sum()
    others = df.T[mask].sum()
    df = df.T[~mask].T
    df['Others'] = others
    df.index = ['Survived','Dead']
    col = df.columns.to_list()
    for count, value in enumerate(col):
        col[count] = feature + '_' + str(value)
    df.columns = col
    df.T.plot(kind='pie', subplots=True, figsize=(20,10), \
        autopct=lambda p:f'{p:.2f}%', title=f'{title}')
    plt.savefig(f'/data/{feature}_relation.png')

def plot_corr():
    result = pd.read_csv('/data/result.csv')
    corr = result.corr()
    corr_series = pd.Series(data = corr.iloc[1,2:])
    corr_sort = corr_series.abs().sort_values(ascending = False)
    plt.figure(figsize=(20,10))
    sns.barplot(corr_sort.values, corr_sort.index, orient='h')
    plt.title("Correlation screening based on correlation coefficient")
    plt.savefig('/data/feature_corrlation.png')

def plot_mutual_info():
    result = pd.read_csv('/data/result.csv')
    result = result.select_dtypes([np.number]).fillna(0)
    mutualInfo = mutual_info_regression(result.iloc[:,2:], result.iloc[:,1], discrete_features='auto')
    mutualInfo_select = pd.Series(data = mutualInfo , index = result.iloc[:,2:].columns).sort_values(ascending = False)
    plt.figure(figsize=(20,10))
    sns.barplot(mutualInfo_select.values, mutualInfo_select.index, orient='h')
    plt.title("Correlation screening based on mutual information")
    plt.savefig('/data/feature_mutual_info.png')
    
def plot_age():
    result = pd.read_csv('/data/result.csv')
    mask = result.Survived == 0
    result.Survived[mask] = 'Dead'
    result.Survived[~mask] = 'Survived'
    sns.set(rc={'figure.figsize':(20,20)})
    facet = sns.FacetGrid(result, hue="Survived",aspect=5)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, result['Age'].max()))
    facet.add_legend()
    plt.title('years old')
    plt.savefig('/data/age_relation.png')

def plot_fare():
    result = pd.read_csv('/data/result.csv')
    mask = result.Survived == 0
    result.Survived[mask] = 'Dead'
    result.Survived[~mask] = 'Survived'
    sns.lineplot(x="Bridge", y="Fare", hue="Survived", style='Sex',
                data=result)
    plt.savefig('/data/bridge_fare.png')
    
if __name__ == '__main__':
    if len(sys.argv) != 2 or (sys.argv[1] != "mutual" and sys.argv[1] != "corr" and sys.argv[1] != "Pclass" and sys.argv[1] != "Bridge" and sys.argv[1] != "Title" and sys.argv[1] != "age" and sys.argv[1] != "sex" and sys.argv[1] != "fare" and sys.argv[1] != "Parch"):
        print(f"Usage: {sys.argv[0]} mutual|corr|Pclass|Bridge|Title|age|sex|fare|Parch")
        exit(1)

    if os.path.exists("/data/result.csv"):
        output = "Figure generated"
    else:
        output = "Result.csv not found, please generate it first"
    command = sys.argv[1]
    if command == "mutual":
        plot_mutual_info()
    elif command == "corr":
        plot_corr()
    elif command == "Pclass":
        bar_chart('Pclass', 'PClass relation with survival')
    elif command == "Bridge":
        pie_chart('Bridge', 'Bridge percentage of survival')
    elif command == "Title":
        pie_chart('Title', 'Title percentage of survival')
    elif command == "age":
        plot_age()
    elif command == "sex":
        bar_chart('Sex', 'Sex relation with survival')
    elif command == "fare":
        plot_fare()
    elif command == "Parch":
        pie_chart('Parch', 'Parch percentage of survival')
    print(yaml.dump({"output":output}))

# Done
