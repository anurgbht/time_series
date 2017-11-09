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


def my_left_join(ref_file,temp_file):
    tt = []
    flag = 1
    for i in range(ref_file.shape[0]):
        for j in range(temp_file.shape[0]):
            if ref_file.iloc[i,0] == temp_file.iloc[j,0]:
##                print(i,j)
                tt.append(temp_file.iloc[j,1])
                flag = 0
                break
        if flag == 1:
            tt.append(None)
        flag = 1
    return tt


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/Strategic Procurement/data/Raw Materials/monthly')

all_files = os.listdir()
target = 'indices_monthly.csv'
all_files.remove(target)
ref_file = pd.read_csv(target)
ref_file = ref_file.iloc[:,[0,1]]
ref_date = make_date(ref_file.iloc[:,0])
ref_x = ref_file.iloc[:,1]
tt = []
reg_dat = []
count = 0
count_reg = 0
col_names = ['date','platts']
for file_name in all_files:
    count += 1
    print(count,file_name)
    dat = pd.read_csv(file_name)
    if dat.shape[0] > 0:
        for j in range(1,dat.shape[1]):
            temp_file = dat.iloc[:,[0,j]]
            tt = my_left_join(ref_file,temp_file)
            ref_file = pd.concat([ref_file,pd.DataFrame(tt)],axis=1)
            my_name = file_name.split('.')[0] + '_' + dat.columns[j]
            my_name = my_name.replace(' ','')
            col_names.append(my_name)

ref_file.columns = col_names    
ref_file.to_csv('for_regression_all_12_10.csv',index=False)
