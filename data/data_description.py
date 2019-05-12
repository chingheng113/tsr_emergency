import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from data import data_util, selected_columns
import seaborn as sns
import matplotlib.pyplot as plt


# data = pd.read_csv(os.path.join('..', 'data', 'tsr_er.csv'))
# data = data_util.get_binary_data(data)
# id_data = data[['ICASE_ID', 'IDCASE_ID']]
# y_data = data[['ICD_ID']]
# X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1)
# id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(id_data, X_data, y_data, test_size=0.33,                                                                       random_state=42)
# # preprocess training data
# id_train, X_train, y_train = data_util.get_binary_Tomek_Links_cleaned_data(id_train, X_train, y_train)
# id_train, X_train, y_train = data_util.get_random_under_samples(id_train, X_train, y_train)
# data_util.save_variables('training_raw.pkl', [id_train, X_train, y_train])

id_train, X_train, y_train = data_util.load_variable(os.path.join('..', 'data', 'training_raw.pkl'))
xy_train = pd.concat([X_train, y_train], axis=1)
# X_train_numeric = X_train.drop(['GENDER_TX'] + selected_columns.dgfa_column, axis=1)
# sns_plot = sns.pairplot(xy_train, vars=['PAD_ID', 'HT_ID'], hue='ICD_ID')
# sns_plot.savefig("output.png")

print(xy_train[(xy_train.PAD_ID == 1.) & (xy_train.ICD_ID == 0.)].shape)
print(xy_train[(xy_train.PAD_ID == 1.) & (xy_train.ICD_ID == 1.)].shape)
print(xy_train[(xy_train.PAD_ID == 0.) & (xy_train.ICD_ID == 0.)].shape)
print(xy_train[(xy_train.PAD_ID == 0.) & (xy_train.ICD_ID == 1.)].shape)