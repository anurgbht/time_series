import os
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pylab as plt
from scipy.stats.stats import pearsonr

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def make_date(dat):
    tt = [datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in dat]
    return tt

def lag_corr(tt1,tt2,n):
    if n > 0:
        temp1 = tt1[:-n]
        temp2 = tt2[n:]
        q = pd.DataFrame([temp1,temp2]).transpose()
        q = q.dropna()
    else:
        q = pd.DataFrame([tt1,tt2]).transpose()
        q = q.dropna()
    return pearsonr(q.iloc[:,0],q.iloc[:,1])[0]
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/Strategic Procurement/data/Raw Materials/monthly')

all_files = os.listdir()
target = 'indices_monthly.csv'
all_files.remove(target)
ref_file = pd.read_csv(target)
ref_date = make_date(ref_file.iloc[:,0])
ref_x = ref_file.iloc[:,1]
tt = []
reg_dat = []
count = 0
count_reg = 0
col_name = []
for file_name in all_files:
    count += 1
    print(count,file_name)
    dat = pd.read_csv(file_name)
    if dat.shape[0] > 0:
        temp_date = make_date(dat.iloc[:,0])
        filt1 = [x in ref_date for x in temp_date]
        filt2 = [x in temp_date for x in ref_date]
        for j in range(1,dat.shape[1]):
            tt3 = []
            q = []
            q.extend([file_name,len(set(ref_date)),len(set(ref_date).intersection(set(temp_date))),len(set(temp_date)),min(ref_date),max(ref_date),min(temp_date),max(temp_date)])
            temp_x = dat.iloc[:,j]
            tt1 = list(temp_x.loc[filt1])
            tt2 = list(ref_x.loc[filt2])
            tt3.append(dat.columns[j])
            tt3.append(lag_corr(tt1,tt2,0))
            tt3.append(lag_corr(tt1,tt2,3))
            q.extend(tt3)
            tt.append(q)

            if ((count_reg == 0) and (len(set(ref_date).intersection(set(temp_date))) == 81)):
                count_reg += 1
                reg_dat.append(tt2[3:])
                col_name.append('platts')
            reg_dat.append(tt1[:-3])
            col_name.append(file_name.replace('.csv','') + '_' + dat.columns[j])
                
            
tt = pd.DataFrame(tt)
reg_dat = pd.DataFrame(reg_dat).transpose()
reg_dat.columns = col_name
reg_dat.to_csv('for_regression_12_10.csv',index=False)
##temp_path = 'D:\OneDrive - Tata Insights and Quants, A division of Tata Industries\Confidential\Projects\Steel\Strategic Procurement/results'
##path = temp_path.replace('\\','/')
##tt.to_csv(path+'/temp.csv',index=False)
