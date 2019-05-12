import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


data = pd.read_csv('mrs_nihss.csv')
data['NIHS_DIFF'] = data['NIHS_TOTAL_OUT'] - data['NIHS_TOTAL_IN']
data['MRS_1_DIFF'] = data['MRS_1']-data['discharged_mrs']
data['MRS_3_DIFF'] = data['MRS_3']-data['discharged_mrs']


# data.boxplot(column=['nihss_total_in'], by='MRS_3')
# plt.title('NIHS_DIFF vs. MRS_3_DIFF')
# bubble_plot(data[['NIHS_DIFF', 'MRS_3_DIFF']], ['NIHS_DIFF', 'MRS_3_DIFF'])


data = data[['NIHS_TOTAL_IN', 'NIHS_TOTAL_OUT', 'discharged_mrs']]
data.dropna(axis=0, inplace=True)
x = [0, 1]
mrs_0 = data[data.discharged_mrs == 0]
in_avg_0 = mrs_0['NIHS_TOTAL_IN'].mean()
in_std_0 = mrs_0['NIHS_TOTAL_IN'].std()
out_avg_0 = mrs_0['NIHS_TOTAL_OUT'].mean()
out_std_0 = mrs_0['NIHS_TOTAL_OUT'].std()
y_0 = [in_avg_0, out_avg_0]
e_0 = [in_std_0, out_std_0]
plt.errorbar(x, y_0, fmt='-o', label='discharged_mrs 0')


mrs_1 = data[data.discharged_mrs == 1]
in_avg_1 = mrs_1['NIHS_TOTAL_IN'].mean()
in_std_1 = mrs_1['NIHS_TOTAL_IN'].std()
out_avg_1 = mrs_1['NIHS_TOTAL_OUT'].mean()
out_std_1 = mrs_1['NIHS_TOTAL_OUT'].std()
y_1 = [in_avg_1, out_avg_1]
e_1 = [in_std_1, out_std_1]
plt.errorbar(x, y_1, fmt='-o', label='discharged_mrs 1')

mrs_2 = data[data.discharged_mrs == 2]
in_avg_2 = mrs_2['NIHS_TOTAL_IN'].mean()
in_std_2 = mrs_2['NIHS_TOTAL_IN'].std()
out_avg_2 = mrs_2['NIHS_TOTAL_OUT'].mean()
out_std_2 = mrs_2['NIHS_TOTAL_OUT'].std()
y_2 = [in_avg_2, out_avg_2]
e_2 = [in_std_2, out_std_2]
plt.errorbar(x, y_2, fmt='-o', label='discharged_mrs 2')

mrs_3 = data[data.discharged_mrs == 3]
in_avg_3 = mrs_3['NIHS_TOTAL_IN'].mean()
in_std_3 = mrs_3['NIHS_TOTAL_IN'].std()
out_avg_3 = mrs_3['NIHS_TOTAL_OUT'].mean()
out_std_3 = mrs_3['NIHS_TOTAL_OUT'].std()
y_3 = [in_avg_3, out_avg_3]
e_3 = [in_std_3, out_std_3]
plt.errorbar(x, y_3, fmt='-o', label='discharged_mrs 3')

mrs_4 = data[data.discharged_mrs == 4]
in_avg_4 = mrs_4['NIHS_TOTAL_IN'].mean()
in_std_4 = mrs_4['NIHS_TOTAL_IN'].std()
out_avg_4 = mrs_4['NIHS_TOTAL_OUT'].mean()
out_std_4 = mrs_4['NIHS_TOTAL_OUT'].std()
y_4 = [in_avg_4, out_avg_4]
e_4 = [in_std_4, out_std_4]
plt.errorbar(x, y_4, fmt='-o', label='discharged_mrs 4')

mrs_5 = data[data.discharged_mrs == 5]
in_avg_5 = mrs_5['NIHS_TOTAL_IN'].mean()
in_std_5 = mrs_5['NIHS_TOTAL_IN'].std()
out_avg_5 = mrs_5['NIHS_TOTAL_OUT'].mean()
out_std_5 = mrs_5['NIHS_TOTAL_OUT'].std()
y_5 = [in_avg_5, out_avg_5]
e_5 = [in_std_5, out_std_5]
plt.errorbar(x, y_5, fmt='-o', label='discharged_mrs 5')

mrs_6 = data[data.discharged_mrs == 6]
in_avg_6 = mrs_6['NIHS_TOTAL_IN'].mean()
in_std_6 = mrs_6['NIHS_TOTAL_IN'].std()
out_avg_6 = mrs_6['NIHS_TOTAL_OUT'].mean()
out_std_6 = mrs_6['NIHS_TOTAL_OUT'].std()
y_6 = [in_avg_6, out_avg_6]
e_6 = [in_std_6, out_std_6]
plt.errorbar(x, y_6, fmt='-o', label='discharged_mrs 6')

plt.xticks([0, 1], labels=['admission', 'discharge'])
plt.ylabel('Average NIHSS total score')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('discharged.png')
plt.show()