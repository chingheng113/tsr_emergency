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
plt.plot(fpr, tpr, label='SVM with DN (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)

fpr, tpr, thresholds = roc_curve(y_test, predict_result['nv'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='SVM with DNV (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


fpr, tpr, thresholds = roc_curve(y_test, predict_result['nvl'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='SVM with DNVL (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


fpr, tpr, thresholds = roc_curve(y_test, predict_result['nvlr'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='SVM with DNVLR (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)


fpr, tpr, thresholds = roc_curve(y_test, predict_result['se'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='SVM with selected features (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)

fpr, tpr, thresholds = roc_curve(y_test, predict_result['lasso'])
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, label='LASSO (AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)

plt.legend(loc="lower right", prop={'size': 12})
plt.savefig('myimage.png', format='png', dpi=1200)
plt.show()