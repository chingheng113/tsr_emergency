import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from data import data_util, selected_columns
from data import selected_columns as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


data = pd.read_csv(os.path.join('..', 'data', 'tsr_er_og.csv'))

case_data = pd.read_csv(os.path.join('..', 'data', 'raw', 'CASEDCASE.csv'))
icd_tex = case_data['ICD_TX'].value_counts()


df_is = data[(data.ICD_ID == 1) | (data.ICD_ID == 2)]
df_he = data[(data.ICD_ID == 3) | (data.ICD_ID == 4)]

data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1, inplace=True)
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True)

# train_data = pd.read_csv(os.path.join('..', 'data', 'training_data_processed.csv'))
# X_train_Lasso = train_data[sc.Lasso_column]
# t_sne = TSNE(n_components=2, perplexity=30).fit_transform(X_train_Lasso)
# # sns.scatterplot(x='tsne-one', y='tsne-two', data=t_sne, c=train_data['ICD_ID'])
# plt.scatter(t_sne[:, 0], t_sne[:, 1], c=train_data['ICD_ID'])

plt.show()