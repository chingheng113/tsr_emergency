import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import selected_columns
from datetime import datetime


def clean_NIHS():
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
    df_nihs['nihs_total_in'] = df_nihs.NIHS_1a_in+df_nihs.NIHS_1b_in+df_nihs.NIHS_1c_in+df_nihs.NIHS_2_in+df_nihs.NIHS_3_in+\
                          df_nihs.NIHS_4_in+df_nihs.NIHS_5aL_in+df_nihs.NIHS_5bR_in+df_nihs.NIHS_6aL_in+\
                          df_nihs.NIHS_6bR_in+df_nihs.NIHS_7_in+df_nihs.NIHS_8_in+df_nihs.NIHS_9_in+df_nihs.NIHS_10_in+\
                          df_nihs.NIHS_11_in
    df_nihs['nihs_total_out'] = df_nihs.NIHS_1a_out+df_nihs.NIHS_1b_out+df_nihs.NIHS_1c_out+df_nihs.NIHS_2_out+df_nihs.NIHS_3_out+ \
                          df_nihs.NIHS_4_out+df_nihs.NIHS_5aL_out+df_nihs.NIHS_5bR_out+df_nihs.NIHS_6aL_out+ \
                          df_nihs.NIHS_6bR_out+df_nihs.NIHS_7_out+df_nihs.NIHS_8_out+df_nihs.NIHS_9_out+df_nihs.NIHS_10_out+ \
                          df_nihs.NIHS_11_out
    df_nihs['diff'] = df_nihs.nihs_total_out - df_nihs.nihs_total_in
    return df_nihs


def get_clean_case_for_er():
    df_case = pd.read_csv('CASEDCASE.csv')
    df_case = df_case[selected_columns.case_column]
    # Replace outlier to Median
    outlier_cols = []
    df_case.loc[df_case.HEIGHT_NM < 100] = np.nan
    df_case.loc[df_case.HEIGHT_NM > 998] = np.nan
    df_case.loc[df_case.WEIGHT_NM > 998] = np.nan
    df_case.loc[df_case.SBP_NM > 998] = np.nan
    df_case.loc[df_case.DBP_NM > 998] = np.nan
    df_case.loc[df_case.BT_NM > 998] = np.nan
    df_case.loc[df_case.HR_NM > 998] = np.nan
    df_case.loc[df_case.RR_NM > 998] = np.nan


    df_case.loc[~df_case.GCSE_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    df_case.loc[~df_case.GCSV_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan
    df_case.loc[~df_case.GCSM_NM.isin(['1', '2', '3', '4', '5', '6'])] = np.nan


def is_tpa():
    df_case = pd.read_csv('CASEDCASE.csv')
    df_tpa = df_case[['IVTPATH_ID', 'IVTPA_DT', 'IVTPAH_NM', 'IVTPAM_NM', 'IVTPAMG_NM', 'NIVTPA_ID']]
    df_tpa.loc[~df_tpa.IVTPATH_ID.isin(['1', '2'])] = np.nan
    df_tpa['IVTPA_DT'] = pd.to_datetime(df_tpa['IVTPA_DT'], format='%Y-%M-%d', errors='ignore')
    df_tpa['NIVTPA_ID'] = [np.nan if np.isnan(x) else -1 for x in df_tpa['IVTPATH_ID']]
    df_case['is_tpa'] = [1 if x == -1 else 0 for x in df_tpa['NIVTPA_ID']]
    # df_forE = df_case[['ICASE_ID', 'IDCASE_ID', 'is_tpa', 'IVTPAMG_NM']]
    # df_forE.to_csv('E.csv')
    return df_case


def tpa_nihs(df_case, df_nihs):
    df = pd.merge(df_case, df_nihs, on=['ICASE_ID', 'IDCASE_ID'])
    df = df[df['is_tpa'] == 1]
    # df = df[['IVTPAMG_NM', 'diff']]
    df = df[['WEIGHT_NM', 'diff']]
    df.dropna(axis=0, inplace=True)
    return df

# def temp_func():
#     df_1 = pd.read_csv('E.csv')
#     df_2 = pd.read_csv('TSR_2018_3m_noMissing_validated.csv')
#     df_2 = df_2[['ICASE_ID', 'IDCASE_ID']]
#     df = pd.merge(df_1, df_2, on=['ICASE_ID', 'IDCASE_ID'])
#     df.to_csv('tpa_hospital.csv')
#     print('a')


if __name__ == '__main__':
    get_clean_case_for_er()
    df_case = is_tpa()

    df_a = df_case[['ORG_ID']]
    df_a.sort_values(by=['ORG_ID'])
    df_a.hist(bins=80)
    plt.show()

    df_nihs = clean_NIHS()
    tpa_nihs = tpa_nihs(df_case, df_nihs)
    plt.figure()
    tpa_nihs.plot.scatter(x='diff', y='WEIGHT_NM')

    plt.figure()
    a = df_nihs[df_nihs['diff'] > 2]
    b = df_nihs[df_nihs['diff'] < -2]
    print(a.shape)
    print(b.shape)
    df_nihs['diff'].hist(bins=84)
    plt.xlabel('Diff NIHSS (discharge - admission)')
    plt.show()
    print('done')