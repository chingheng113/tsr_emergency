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
    # 'GCSE_NM',
    # 'GCSV_NM',
    # 'GCSM_NM',
    'SBP_NM',
    'DBP_NM'
    # 'BT_NM',
    # 'HR_NM',
    # 'RR_NM'
]

lb_column = [
    # 'HB_NM',
    # 'HCT_NM',
    'PLATELET_NM',
    # 'WBC_NM',
    'PTT1_NM',
    # 'PTT2_NM',
    'PTINR_NM',
    'ER_NM',
    # 'BUN_NM',
    'CRE_NM',
    # 'ALB_NM',
    # 'CRP_NM',
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
    # 'UR_ID',
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


ElasticNet_drop_column = [
            'HCT_NM',
            'NIHS_4_in',
            'PTINR_NM',
            'NIHS_9_in',
            'ER_NM',
            'CRE_NM',
            'GCSM_NM',
            'CRP_NM',
            'GENDER_0',
            'GENDER_1',
            'NIHS_3_in',
            'ALB_NM',
            'NIHS_11_in',
            'HR_NM',
            'BT_NM',
            'PTT1_NM',
            'UR_ID',
            'RR_NM'
]