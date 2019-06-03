import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from data import data_util, selected_columns
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(os.path.join('..', 'data', 'tsr_er_og.csv'))
df_is = data[(data.ICD_ID == 1) | (data.ICD_ID == 2)]
df_he = data[(data.ICD_ID == 3) | (data.ICD_ID == 4)]

# print(data.shape[0])
# print(df_is.shape[0])
# print(df_he.shape[0])
# print(data[data.GENDER_TX == 0].shape[0])
# print(data[data.GENDER_TX == 0].shape[0]/data.shape[0])
#
# print(df_is[df_is.GENDER_TX == 0].shape[0])
# print(df_is[df_is.GENDER_TX == 0].shape[0]/df_is.shape[0])
#
# print(df_he[df_he.GENDER_TX == 0].shape[0])
# print(df_he[df_he.GENDER_TX == 0].shape[0]/df_he.shape[0])
a='HBAC_NM'
# print(data[a].dropna().shape[0])
# print(data[a].dropna().mean(), data[a].dropna().std())
# print(df_is[a].dropna().shape[0])
# print(df_is[a].dropna().mean(), df_is[a].dropna().std())
# print(df_he[a].dropna().shape[0])
# print(df_he[a].dropna().mean(), df_he[a].dropna().std())

b ='HT_ID'
print(data[data.NIHS_1a_in ==0].shape[0])