from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join('/data/linc9/tsr_er/')))
from data import data_util, selected_columns

training_data_name = os.path.join('..', 'data', 'training_data.pkl')
testing_data_name = os.path.join('..', 'data', 'testing_data.pkl')
if os.path.isfile(training_data_name) & os.path.isfile(testing_data_name):
    id_train, X_train, y_train = data_util.load_variable(training_data_name)
    # train_df = pd.concat([id_train, X_train, y_train], axis=1)
    # train_df.to_csv('training_data.csv', index=False)
    id_test, X_test, y_test = data_util.load_variable(testing_data_name)
    # test_df = pd.concat([id_test, X_test, y_test], axis=1)
    # test_df.to_csv('testing_data.csv', index=False)
else:
    data = pd.read_csv(os.path.join('..', 'data', 'tsr_er.csv'))
    data = data_util.get_binary_data(data)
    data = shuffle(data)

    id_data = data[['ICASE_ID', 'IDCASE_ID']]
    y_data = data[['ICD_ID']]
    X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1)
    id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(id_data, X_data, y_data, test_size=0.33, random_state=42)
    # preprocess training data
    id_train, X_train, y_train = data_util.get_binary_Tomek_Links_cleaned_data(id_train, X_train, y_train)
    id_train, X_train, y_train = data_util.get_random_under_samples(id_train, X_train, y_train)
    scaler, X_train = data_util.normalization_onehotcoding_for_training(X_train)
    # preprocess testing data
    X_test = data_util.normalization_onehotcoding_for_testing(X_test, scaler)
    # saving data
    data_util.save_variables(training_data_name, [id_train, X_train, y_train])
    data_util.save_variables(testing_data_name, [id_test, X_test, y_test])


clf = RandomForestClassifier(n_estimators=1000, random_state=123, class_weight='balanced')
# clf = SVC(probability=True)
# All Features
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(classification_report(y_test, y_predict))
mc = confusion_matrix(y_test, y_predict)
print(mc)
y_probas = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_probas[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)

# == Feature importance
feature_names = X_train.columns.values
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(X_train.shape[1]):
    print("%d. %f %s" % (i + 1, importances[indices[i]], feature_names[indices[i]]))
# ===

# ElasticNet Features
X_train_Lasso = X_train.drop(selected_columns.ElasticNet_drop_column, axis=1)
X_test_Lasso = X_test.drop(selected_columns.ElasticNet_drop_column, axis=1)
clf.fit(X_train_Lasso, y_train)
y_predict = clf.predict(X_test_Lasso)
print(classification_report(y_test, y_predict))
mc = confusion_matrix(y_test, y_predict)
print(mc)
y_probas = clf.predict_proba(X_test_Lasso)
fpr, tpr, thresholds = roc_curve(y_test, y_probas[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)

# no Risk Factors
X_train_no_dgfa = X_train.drop(selected_columns.dgfa_column, axis=1)
clf.fit(X_train_no_dgfa, y_train)
X_test_no_dgfa = X_test.drop(selected_columns.dgfa_column, axis=1)
y_predict_no_dgfa = clf.predict(X_test_no_dgfa)
print(classification_report(y_test, y_predict_no_dgfa))
mc = confusion_matrix(y_test, y_predict_no_dgfa)
print(mc)
y_probas_no_dgfa = clf.predict_proba(X_test_no_dgfa)
fpr, tpr, thresholds = roc_curve(y_test, y_probas_no_dgfa[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)