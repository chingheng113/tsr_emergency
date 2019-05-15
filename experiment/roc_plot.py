import pandas as pd
from sklearn.metrics import roc_curve, auc
from data import data_util
import os
import matplotlib.pyplot as plt


predict_result = pd.read_csv('predict_result.csv')
testing_data_name = os.path.join('..', 'data', 'testing_data_processed.pkl')
id_test, X_test, y_test = data_util.load_variable(testing_data_name)

fpr, tpr, thresholds = roc_curve(y_test, predict_result['n'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='NIHSS (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)

fpr, tpr, thresholds = roc_curve(y_test, predict_result['nv'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='NIHSS+VS (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


fpr, tpr, thresholds = roc_curve(y_test, predict_result['nvl'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='NIHSS+VS+LB (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


fpr, tpr, thresholds = roc_curve(y_test, predict_result['nvlr'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='NIHSS+VS+LB+RF (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


fpr, tpr, thresholds = roc_curve(y_test, predict_result['se'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='Elastic features (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


plt.legend(loc="lower right", prop={'size': 12})
plt.show()