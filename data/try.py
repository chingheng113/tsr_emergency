import pandas as pd
import matplotlib.pyplot as plt

def bubble_plot(data, group_names):
    data_size = data.groupby(group_names).size()
    keys = data.groupby(group_names).groups.keys()
    x =[]
    y =[]
    for key in keys:
        x.append(key[0])
        y.append(key[1])
    plt.scatter(x, y, s=data_size, alpha=.5)
    plt.xlabel(group_names[0])
    plt.ylabel(group_names[1])


data = pd.read_csv('TSR_2018_3m_noMissing_validated.csv')
data['nihss_total_in'] = data.NIHS_1a_in+data.NIHS_1b_in+data.NIHS_1c_in+data.NIHS_2_in+data.NIHS_3_in+\
                          data.NIHS_4_in+data.NIHS_5aL_in+data.NIHS_5bR_in+data.NIHS_6aL_in+\
                          data.NIHS_6bR_in+data.NIHS_7_in+data.NIHS_8_in+data.NIHS_9_in+data.NIHS_10_in+\
                          data.NIHS_11_in
mrs_3 = data['MRS_3']

# data.boxplot(column=['nihss_total_in'], by='MRS_3')
bubble_plot(data[['nihss_total_in', 'discharged_mrs']], ['nihss_total_in', 'discharged_mrs'])
plt.savefig('a.png')
plt.show()