id_column = [
    'ICASE_ID',
    'IDCASE_ID'
]

dm_column = [
    'ORG_ID',
    'HEIGHT_NM',
    'WEIGHT_NM',
    'OPC_ID',
    'ONSET_DT'
]

gcs_vital_column = [
    'SBP_NM',
    'DBP_NM'
]

lb_column = [
    'PLATELET_NM',
    'PTT1_NM',
    'PTINR_NM',
    'ER_NM',
    'CRE_NM',
    'HBAC_NM'
]

case_column = id_column+dm_column+gcs_vital_column+lb_column+['ICD_ID']

nihs_column = [
    'NIHS_1a_in',
    'NIHS_1b_in',
    'NIHS_1c_in',
    'NIHS_2_in',
    'NIHS_3_in',
    'NIHS_4_in',
    'NIHS_5aL_in',
    'NIHS_5bR_in',
    'NIHS_6aL_in',
    'NIHS_6bR_in',
    'NIHS_7_in',
    'NIHS_8_in',
    'NIHS_9_in',
    'NIHS_10_in',
    'NIHS_11_in'
]

dgfa_column = [
    'HT_ID',
    'HC_ID',
    'DM_ID',
    'PCVA_ID',
    'PTIA_ID',
    'HD_ID',
    'PAD_ID'
]

dgfa_column_converted = [
    'HT_ID',
    'HC_ID',
    'DM_ID',
    'PISCH_ID',
    'HD_ID',
    'PAD_ID'
]

case_column_date = [
            'ONSETH_NM',
            'ONSETM_NM',
            'OT_DT',
            'OTTIH_NM',
            'OTTIM_NM',
            'FLOOK_DT',
            'FLOOKH_NM',
            'FLOOKM_NM',
            'NIHSIN_DT',
            'NIHSINH_NM',
            'NIHSINM_NM'
]


Lasso_column = [
    "HEIGHT_NM",
    "WEIGHT_NM",
    "SBP_NM",
    "DBP_NM",
    "PLATELET_NM",
    "PTT1_NM",
    "PTINR_NM",
    "ER_NM",
    "CRE_NM",
    "NIHS_1a_in",
    "NIHS_1b_in",
    "NIHS_2_in",
    "NIHS_4_in",
    "NIHS_5aL_in",
    "NIHS_5bR_in",
    "NIHS_6aL_in",
    "NIHS_6bR_in",
    "NIHS_7_in",
    "NIHS_10_in",
    "NIHS_11_in",
    "AGE",
    "HT_ID",
    "HC_ID",
    "DM_ID",
    "PISCH_ID",
    "HD_ID",
    "PAD_ID"
]