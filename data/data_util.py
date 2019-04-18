import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import selected_columns
from sklearn.preprocessing import Imputer
from scipy.stats import iqr
from datetime import datetime


def clean_nihs():
    df_nihs = pd.read_csv('CASEDNIHS(denormalized).csv')
    # print(df_nihs.shape)
    df_nihs.loc[~df_nihs.NIHS_1a_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_1a_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_1b_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_1b_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_1c_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_1c_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_2_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_2_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_3_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_3_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_4_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_4_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_5aL_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_5aL_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_5bR_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_5bR_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_6aL_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_6aL_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_6bR_in.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_6bR_out.isin(['0', '1', '2', '3', '4'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_7_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_7_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_8_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_8_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_9_in.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_9_out.isin(['0', '1', '2', '3'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_10_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_10_out.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_11_in.isin(['0', '1', '2'])] = np.nan
    df_nihs.loc[~df_nihs.NIHS_11_out.isin(['0', '1', '2'])] = np.nan
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
    df_case['GCSE_NM'].loc[~df_case.GCSE_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    df_case['GCSV_NM'].loc[~df_case.GCSV_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    df_case['GCSM_NM'].loc[~df_case.GCSM_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    # Date type
    df_case['ONSET_DT'] = pd.to_datetime(df_case['ONSET_DT'], format='%Y-%M-%d', errors='ignore')
    df_case.dropna(axis=0)
    return df_case


if __name__ == '__main__':
    df_case = get_cleaned_case_for_er()
    df_nihs = get_cleaned_nihs_for_er()
    a = 1