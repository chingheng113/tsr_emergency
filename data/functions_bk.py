import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import selected_columns
from sklearn.preprocessing import Imputer


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


def temp_func():
    df_1 = pd.read_csv('E.csv')
    df_2 = pd.read_csv('TSR_2018_3m_noMissing_validated.csv')
    df_2 = df_2[['ICASE_ID', 'IDCASE_ID']]
    df = pd.merge(df_1, df_2, on=['ICASE_ID', 'IDCASE_ID'])
    df.to_csv('tpa_hospital.csv')
