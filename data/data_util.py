import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import selected_columns
from sklearn.preprocessing import Imputer
from sklearn.utils import resample
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import TomekLinks



def clean_nihs():
    df_nihs = pd.read_csv('CASEDNIHS(denormalized).csv')
    # print(df_nihs.shape)
    df_nihs['NIHS_1a_in'].loc[~df_nihs.NIHS_1a_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_1a_out'].loc[~df_nihs.NIHS_1a_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_1b_in'].loc[~df_nihs.NIHS_1b_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_1b_out'].loc[~df_nihs.NIHS_1b_out.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_1c_in'].loc[~df_nihs.NIHS_1c_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_1c_out'].loc[~df_nihs.NIHS_1c_out.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_2_in'].loc[~df_nihs.NIHS_2_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_2_out'].loc[~df_nihs.NIHS_2_out.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_3_in'].loc[~df_nihs.NIHS_3_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_3_out'].loc[~df_nihs.NIHS_3_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_4_in'].loc[~df_nihs.NIHS_4_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_4_out'].loc[~df_nihs.NIHS_4_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_5aL_in'].loc[~df_nihs.NIHS_5aL_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_5aL_out'].loc[~df_nihs.NIHS_5aL_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_5bR_in'].loc[~df_nihs.NIHS_5bR_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_5bR_out'].loc[~df_nihs.NIHS_5bR_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_6aL_in'].loc[~df_nihs.NIHS_6aL_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_6aL_out'].loc[~df_nihs.NIHS_6aL_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_6bR_in'].loc[~df_nihs.NIHS_6bR_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_6bR_out'].loc[~df_nihs.NIHS_6bR_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs['NIHS_7_in'].loc[~df_nihs.NIHS_7_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_7_out'].loc[~df_nihs.NIHS_7_out.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_8_in'].loc[~df_nihs.NIHS_8_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_8_out'].loc[~df_nihs.NIHS_8_out.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_9_in'].loc[~df_nihs.NIHS_9_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_9_out'].loc[~df_nihs.NIHS_9_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs['NIHS_10_in'].loc[~df_nihs.NIHS_10_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_10_out'].loc[~df_nihs.NIHS_10_out.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_11_in'].loc[~df_nihs.NIHS_11_in.isin(['0', '1', '2'])] = np.nan
    df_nihs['NIHS_11_out'].loc[~df_nihs.NIHS_11_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.dropna(axis=0, inplace=True)
    # print(df_nihs.shape)
    df_nihs['NIHS_TOTAL_IN'] = df_nihs.NIHS_1a_in+df_nihs.NIHS_1b_in+df_nihs.NIHS_1c_in+df_nihs.NIHS_2_in+df_nihs.NIHS_3_in+\
                          df_nihs.NIHS_4_in+df_nihs.NIHS_5aL_in+df_nihs.NIHS_5bR_in+df_nihs.NIHS_6aL_in+\
                          df_nihs.NIHS_6bR_in+df_nihs.NIHS_7_in+df_nihs.NIHS_8_in+df_nihs.NIHS_9_in+df_nihs.NIHS_10_in+\
                          df_nihs.NIHS_11_in
    df_nihs['NIHS_TOTAL_OUT'] = df_nihs.NIHS_1a_out+df_nihs.NIHS_1b_out+df_nihs.NIHS_1c_out+df_nihs.NIHS_2_out+df_nihs.NIHS_3_out+ \
                          df_nihs.NIHS_4_out+df_nihs.NIHS_5aL_out+df_nihs.NIHS_5bR_out+df_nihs.NIHS_6aL_out+ \
                          df_nihs.NIHS_6bR_out+df_nihs.NIHS_7_out+df_nihs.NIHS_8_out+df_nihs.NIHS_9_out+df_nihs.NIHS_10_out+ \
                          df_nihs.NIHS_11_out
    df_nihs['DIFF'] = df_nihs.NIHS_TOTAL_OUT - df_nihs.NIHS_TOTAL_IN
    return df_nihs


def clean_dbmrs():
    df_dbmrs = pd.read_csv('CASEDBMRS(denormalized).csv')
    df_dbmrs['Feeding'].loc[~df_dbmrs.Feeding.isin(['0', '5', '10'])] = np.nan
    df_dbmrs['Transfers'].loc[~df_dbmrs.Transfers.isin(['0', '5', '10', '15'])] = np.nan
    df_dbmrs['Bathing'].loc[~df_dbmrs.Bathing.isin(['0', '5'])] = np.nan
    df_dbmrs['Toilet_use'].loc[~df_dbmrs.Toilet_use.isin(['0', '5', '10'])] = np.nan
    df_dbmrs['Grooming'].loc[~df_dbmrs.Grooming.isin(['0', '5'])] = np.nan
    df_dbmrs['Mobility'].loc[~df_dbmrs.Mobility.isin(['0', '5', '10', '15'])] = np.nan
    df_dbmrs['Stairs'].loc[~df_dbmrs.Stairs.isin(['0', '5', '10'])] = np.nan
    df_dbmrs['Dressing'].loc[~df_dbmrs.Dressing.isin(['0', '5', '10'])] = np.nan
    df_dbmrs['Bowel_control'].loc[~df_dbmrs.Bowel_control.isin(['0', '5', '10'])] = np.nan
    df_dbmrs['Bladder_control'].loc[~df_dbmrs.Bladder_control.isin(['0', '5', '10'])] = np.nan
    df_dbmrs['discharged_mrs'].loc[~df_dbmrs.discharged_mrs.isin(['0', '1', '2', '3', '4', '5', '6'])] = np.nan
    return df_dbmrs


def clean_rfur():
    df_rfur = pd.read_csv('CASEDRFUR(denormalized).csv')
    rfur_cols = ['VERS_1', 'VERS_3', 'VERS_6', 'VERS_12', 'VEIHD_1', 'VEIHD_3', 'VEIHD_6', 'VEIHD_12']
    df_rfur.drop(rfur_cols, axis=1, inplace=True)
    df_rfur['MRS_1'].loc[~df_rfur.MRS_1.isin(['0', '1', '2', '3', '4', '5', '6'])] = np.nan
    df_rfur['MRS_3'].loc[~df_rfur.MRS_3.isin(['0', '1', '2', '3', '4', '5', '6'])] = np.nan
    df_rfur['MRS_6'].loc[~df_rfur.MRS_6.isin(['0', '1', '2', '3', '4', '5', '6'])] = np.nan
    df_rfur['MRS_12'].loc[~df_rfur.MRS_12.isin(['0', '1', '2', '3', '4', '5', '6'])] = np.nan
    return df_rfur


def get_cleaned_dgfa_for_er():
    df_dgfa = pd.read_csv('CASEDDGFA.csv')
    df_dgfa = df_dgfa[['ICASE_ID', 'IDCASE_ID']+selected_columns.dgfa_column]
    for c in selected_columns.dgfa_column:
        # Don't want 'don't know'
        df_dgfa[c].loc[~df_dgfa[c].isin([0., 1.])] = np.nan
    df_dgfa.dropna(axis=0, inplace=True)
    return df_dgfa

def get_cleaned_nihs_for_er():
    df_nihs = clean_nihs()
    df_nihs = df_nihs[selected_columns.nihs_column]
    df_nihs.dropna(axis=0)
    return df_nihs


def outliers_iqr(ys):
    # http://colingorrie.github.io/outlier-detection.html
    quartile_1, quartile_3 = np.nanpercentile(ys, [25, 75],)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))


def outlier_to_mean(df, columns):
    for col in columns:
        df[col].loc[df[col] > 998] = np.nan
        outlier_inx = outliers_iqr(df[col])
        df[col].loc[outlier_inx] = np.nan
    df[columns] = Imputer(missing_values=np.nan, strategy='mean', axis=0).fit_transform(df[columns])
    return df


def get_cleaned_case_for_er():
    df_case = pd.read_csv('CASEDCASE.csv')
    df_case = df_case[selected_columns.case_column]

    # Replace numeric outlier to Median
    outlier_cols = ['HEIGHT_NM', 'WEIGHT_NM', 'SBP_NM', 'DBP_NM', 'BT_NM', 'HR_NM', 'RR_NM',
                    'HB_NM', 'HCT_NM', 'PLATELET_NM', 'WBC_NM', 'PTT1_NM', 'PTT2_NM', 'PTINR_NM',
                    'ER_NM', 'BUN_NM', 'CRE_NM', 'ALB_NM', 'CRP_NM', 'HBAC_NM']
    df_case = outlier_to_mean(df_case, outlier_cols)
    # Degree type data
    df_case['OPC_ID'].loc[~df_case.OPC_ID.isin(['1', '2', '3'])] = np.nan
    df_case['ICD_ID'].loc[~df_case.ICD_ID.isin(['1', '2', '3', '4'])] = np.nan
    df_case['GCSE_NM'].loc[~df_case.GCSE_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    df_case['GCSV_NM'].loc[~df_case.GCSV_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    df_case['GCSM_NM'].loc[~df_case.GCSM_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    # Date type
    df_case['ONSET_DT'] = pd.to_datetime(df_case['ONSET_DT'], format='%Y-%M-%d', errors='coerce')
    df_case.dropna(axis=0, inplace=True)
    return df_case


def get_cleaned_mcase():
    df_mcase = pd.read_csv('CASEMCASE.csv')
    df_mcase = df_mcase[['ICASE_ID', 'BIRTH_DT', 'GENDER_TX']]
    df_mcase['BIRTH_DT'] = pd.to_datetime(df_mcase['BIRTH_DT'], format='%Y-%M-%d', errors='coerce')
    df_mcase['GENDER_TX'] = df_mcase['GENDER_TX'].replace(to_replace={'F': 0, 'M': 1})
    return df_mcase


def get_age(df):
    age = df['ONSET_DT'].year - df['BIRTH_DT'].year - ((df['ONSET_DT'].month, df['ONSET_DT'].day) < (df['BIRTH_DT'].month, df['BIRTH_DT'].day))
    return age


def calculate_age(df):
    age = df.apply(get_age, axis=1)
    df['AGE'] = age
    df.drop(['BIRTH_DT', 'ONSET_DT'], axis=1, inplace=True)
    return df


def exclusion_criteria(df):
    # only emergency people
    df = df[df.OPC_ID == 3]
    # we don't need these feature for classification
    df.drop(['ORG_ID', 'OPC_ID'], axis=1, inplace=True)
    return df


def normalization_onehotcoding_for_training(X_df):
    scaler = StandardScaler()
    X_data_category = pd.get_dummies(X_df['GENDER_TX'].astype(int), prefix='GENDER')
    x_data_bool = X_df[selected_columns.dgfa_column]
    X_data_numeric = X_df.drop(['GENDER_TX']+selected_columns.dgfa_column, axis=1)
    scaler.fit(X_data_numeric)
    X_data_scaled = scaler.transform(X_data_numeric)
    X_data_scaled = pd.DataFrame(X_data_scaled, index=X_data_numeric.index, columns=X_data_numeric.columns)
    result = pd.concat([X_data_category, X_data_scaled, x_data_bool], axis=1)
    return scaler, result


def normalization_onehotcoding_for_testing(X_df, scaler):
    X_data_category = pd.get_dummies(X_df['GENDER_TX'].astype(int), prefix='GENDER')
    x_data_bool = X_df[selected_columns.dgfa_column]
    X_data_numeric = X_df.drop(['GENDER_TX'] + selected_columns.dgfa_column, axis=1)
    X_data_scaled = scaler.transform(X_data_numeric)
    X_data_scaled = pd.DataFrame(X_data_scaled, index=X_data_numeric.index, columns=X_data_numeric.columns)
    result = pd.concat([X_data_category, X_data_scaled, x_data_bool], axis=1)
    return result


def get_multi_balanced_data(df):
    df_1 = df[df.ICD_ID == 1]
    print(df_1.shape)
    df_2 = df[df.ICD_ID == 2]
    print(df_2.shape)
    df_3 = df[df.ICD_ID == 3]
    print(df_3.shape)
    df_4 = df[df.ICD_ID == 4]
    print(df_4.shape)

    resample_size = min([df_1.shape[0], df_2.shape[0], df_3.shape[0], df_4.shape[0]])
    df_1_downsampled = resample(df_1, replace=False,  # sample without replacement
                                n_samples=resample_size,  # to match minority class
                                random_state=7)  # reproducible results
    df_2_downsampled = resample(df_2, replace=False,  # sample without replacement
                                n_samples=resample_size,  # to match minority class
                                random_state=7)  # reproducible results
    df_3_downsampled = resample(df_3, replace=False,  # sample without replacement
                                n_samples=resample_size,  # to match minority class
                                random_state=7)  # reproducible results
    df_4_downsampled = resample(df_4, replace=False,  # sample without replacement
                                n_samples=resample_size,  # to match minority class
                                random_state=7)  # reproducible results
    result = pd.concat([df_1_downsampled, df_2_downsampled, df_3_downsampled, df_4_downsampled], axis=0)
    return result


def get_binary_Tomek_Links_cleaned_data(id_df, X_df, y_df):
    tLinks = TomekLinks()
    tLinks.fit_sample(X_df, y_df)
    sample_indices = tLinks.sample_indices_
    id_df_cleaned = id_df.iloc[sample_indices]
    X_df_cleaned = X_df.iloc[sample_indices]
    y_df_cleaned = y_df.iloc[sample_indices]
    return id_df_cleaned, X_df_cleaned, y_df_cleaned


def get_binary_data(df):
    # ischemic: ICD_ID == 1, 2
    # hemorrhagic: ICD_ID == 3, 4
    df['ICD_ID'] = df['ICD_ID'].replace(to_replace={1: 0, 2: 0, 3: 1, 4: 1})
    return df


def create_tsr_er_dataset():
    df_case = get_cleaned_case_for_er()
    df_nihs = get_cleaned_nihs_for_er()
    df_mcase = get_cleaned_mcase()
    df_dgfr = get_cleaned_dgfa_for_er()
    df_merged = pd.merge(df_case, df_nihs, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_dgfr, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_mcase, on=['ICASE_ID'])
    df_merged.dropna(axis=0, inplace=True)
    df_result = calculate_age(df_merged)
    df_result = exclusion_criteria(df_result)
    df_result.to_csv('tsr_er.csv', index=False)


def create_mrs_nih_dataset():
    df_mcase = get_cleaned_mcase()
    df_dbmrs = clean_dbmrs() # still keep null
    df_nihs = clean_nihs() # still keep null
    df_rfur = clean_rfur() # still keep null
    df_merged = pd.merge(df_nihs, df_dbmrs, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_rfur, on=['ICASE_ID', 'IDCASE_ID'])
    df_merged = pd.merge(df_merged, df_mcase, on=['ICASE_ID'])
    df_merged.to_csv('mrs_nihss.csv', index=False)


if __name__ == '__main__':
    create_tsr_er_dataset()
    # create_mrs_nih_dataset()
