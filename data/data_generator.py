import pandas as pd
import numpy as np
from data import data_util
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from data import selected_columns
current_path = os.path.dirname(__file__)


def create_tsr_er_dataset():
    df_case = data_util.get_cleaned_case_for_er()
    df_nihs = data_util.get_cleaned_nihs_for_er()
    df_mcase = data_util.get_cleaned_mcase()
    df_dgfr = data_util.get_cleaned_dgfa_for_er()
    df_merged = pd.merge(df_case, df_nihs, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_dgfr, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_mcase, on=['ICASE_ID'])
    df_merged.dropna(axis=0, inplace=True)
    df_result = data_util.calculate_age(df_merged)
    df_result = data_util.exclusion_criteria(df_result)
    df_result.to_csv('tsr_er_og.csv', index=False)


def create_mrs_nih_dataset():
    df_mcase = data_util.get_cleaned_mcase()
    df_dbmrs = data_util.clean_dbmrs() # still keep null
    df_nihs = data_util.clean_nihs() # still keep null
    df_rfur = data_util.clean_rfur() # still keep null
    df_merged = pd.merge(df_nihs, df_dbmrs, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_rfur, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_mcase, on=['ICASE_ID'])
    df_merged.to_csv('mrs_nihss.csv', index=False)


def replace_outlier_to_median(data):
    data_numeric = data.drop(['GENDER_TX']+['ICD_ID']
                             + selected_columns.dgfa_column
                             + selected_columns.id_column
                             + selected_columns.nihs_column
                             , axis=1)
    data = data_util.outlier_to_mean(data, data_numeric.columns)
    return data


def get_training_testing_data():
    data = pd.read_csv(os.path.join(current_path, 'tsr_er_og.csv'))
    data = data_util.get_binary_data(data)
    data = shuffle(data)
    data = replace_outlier_to_median(data)
    id_data = data[['ICASE_ID', 'IDCASE_ID']]
    y_data = data[['ICD_ID']]
    X_data = data.drop(['ICASE_ID', 'IDCASE_ID', 'ICD_ID'], axis=1)
    id_train, id_test, X_train, X_test, y_train, y_test = train_test_split(id_data, X_data, y_data, test_size=0.33,
                                                                           random_state=42)

    train_df = pd.concat([id_train, X_train, y_train], axis=1)
    train_df.to_csv(os.path.join('..', 'data', 'training_data_og.csv'), index=False)
    test_df = pd.concat([id_test, X_test, y_test], axis=1)
    test_df.to_csv(os.path.join('..', 'data', 'testing_data_og.csv'), index=False)

    # preprocess training data
    id_train, X_train, y_train = data_util.get_binary_Tomek_Links_cleaned_data(id_train, X_train, y_train)
    id_train, X_train, y_train = data_util.get_random_under_samples(id_train, X_train, y_train)
    scaler, X_train = data_util.normalization_onehotcoding_for_training(X_train)
    # preprocess testing data
    X_test = data_util.normalization_onehotcoding_for_testing(X_test, scaler)
    return id_train, id_test, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    create_tsr_er_dataset()
    # get_training_testing_data()