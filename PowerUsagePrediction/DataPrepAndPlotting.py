#!/usr/bin/env python
# coding: utf-8

# In[31]:

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt

df_9092 = pd.read_excel('data/9798.xlsx', sheet_name='مناطق', header=4)  
df_9092.head()
df_9092.info()
df_9092.describe()
df_9092['تاریخ'] = df_9092['تاریخ'].str.replace('/','-')
df_9092 = df_9092.set_index("تاریخ")
df_9092.replace(to_replace=0,  method='ffill', inplace=True)
df_9092.tail()
spring_90 = df_9092.loc['1398-01-01':'1398-03-31']
summer_90 = df_9092.loc['1398-04-01':'1398-06-31']
fall_90 = df_9092.loc['1398-07-01':'1398-09-30']
winter_90 = df_9092.loc['1398-10-01':'1398-12-30']
df_l = [spring_90, summer_90, fall_90, winter_90]
season_l = ['Spring', 'Summer', 'Fall', 'Winter', 'Winter']
for i in df_l:
    sns.set_theme(style="darkgrid")
    plt.figure(figsize = (40,10))
    sns.lineplot(data=i)
    plt.xticks(rotation=30)
    plt.xlabel("Date") 
    plt.ylabel("Usage(MWH)")
    for s in season_l:
        plt.title(s+' | 1398')
        plt.savefig(s+"-1398.jpg")
        season_l.pop(0)
        break
    print("Done!")