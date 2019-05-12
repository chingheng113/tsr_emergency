from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from data import data_util

data = pd.read_csv(os.path.join('..', 'data', 'tsr_er.csv'))
data = data_util.get_multi_balanced_data(data)

data = shuffle(data)
id_data = data[['ICASE_ID', 'IDCASE_ID']]

y_data = data[['ICD_ID']]

X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1)


id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(id_data, X_data, y_data, test_size=0.33, random_state=42)
# clf = LinearSVC(multi_class='crammer_singer')
clf = RandomForestClassifier(n_estimators=100, random_state=123)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(classification_report(y_test, y_predict))

# try...
# https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html#sphx-glr-auto-examples-multioutput-plot-classifier-chain-yeast-py