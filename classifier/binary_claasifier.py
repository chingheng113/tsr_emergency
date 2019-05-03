from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join('/data/linc9/tsr_er/')))
from data import data_util, selected_columns

training_data_name = os.path.join('..', 'data', 'training_data.pkl')
testing_data_name = os.path.join('..', 'data', 'testing_data.pkl')
if os.path.isfile(training_data_name) & os.path.isfile(testing_data_name):
    id_train, X_train, y_train = data_util.load_variable(training_data_name)
    id_test, X_test, y_test = data_util.load_variable(testing_data_name)
else:
    data = pd.read_csv(os.path.join('..', 'data', 'tsr_er.csv'))
    data = data_util.get_binary_data(data)
    data = shuffle(data)

    id_data = data[['ICASE_ID', 'IDCASE_ID']]
    y_data = data[['ICD_ID']]
    X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1)
    id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(id_data, X_data, y_data, test_size=0.33, random_state=42)
    # preprocessing
    scaler, X_train = data_util.normalization_onehotcoding_for_training(X_train)
    print(y_train[y_train.ICD_ID == 0].shape[0]/y_train.shape[0])
    id_train, X_train, y_train = data_util.get_binary_Tomek_Links_cleaned_data(id_train, X_train, y_train)
    print(y_train[y_train.ICD_ID == 0].shape[0]/y_train.shape[0])
    X_test = data_util.normalization_onehotcoding_for_testing(X_test, scaler)
    # saving data
    data_util.save_variables(training_data_name, [id_train, X_train, y_train])
    data_util.save_variables(testing_data_name, [id_test, X_test, y_test])


# clf = SVC(kernel='rbf', random_state=42, class_weight='balanced')
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=123)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(classification_report(y_test, y_predict))
mc = confusion_matrix(y_test, y_predict)
print(mc)

X_train_no_dgfa = X_train.drop(selected_columns.dgfa_column, axis=1)
clf.fit(X_train_no_dgfa, y_train)
X_test_no_dgfa = X_test.drop(selected_columns.dgfa_column, axis=1)
y_predict_no_dgfa = clf.predict(X_test_no_dgfa)
print(classification_report(y_test, y_predict_no_dgfa))
mc = confusion_matrix(y_test, y_predict_no_dgfa)
print(mc)