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

data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1, inplace=True)
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True)
plt.show()