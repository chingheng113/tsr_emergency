from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join('/data/linc9/tsr_er/')))
from data import data_util, data_generator
import matplotlib.pyplot as plt


training_data_name = os.path.join('..', 'data', 'training_data_processed.pkl')
testing_data_name = os.path.join('..', 'data', 'testing_data_processed.pkl')
if os.path.isfile(training_data_name) & os.path.isfile(testing_data_name):
    id_train, X_train, y_train = data_util.load_variable(training_data_name)
    id_test, X_test, y_test = data_util.load_variable(testing_data_name)
else:
    id_train, id_test, X_train, X_test, y_train, y_test = data_generator.get_training_testing_data()
    # saving data
    data_util.save_variables(training_data_name, [id_train, X_train, y_train])
    data_util.save_variables(testing_data_name, [id_test, X_test, y_test])

forest = ExtraTreesClassifier()
forest.fit(X_train, y_train)
# print(model.feature_importances_)
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
col_name = X_train.columns
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, col_name[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

print('done')
