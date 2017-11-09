import os
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pylab as plt
from statsmodels.tsa.stattools import acf, pacf,adfuller
import arch.unitroot as auni
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def agg_time(time):
    month_time = [x.month for x in time]
    year_time = [x.year for x in time]
    ret_tt_time = []
    for i in range(len(time)-1):
        if month_time[i] != month_time[i+1]:
            if i != len(time)-2:
                ret_tt_time.append(datetime.strptime((str(month_time[i])+'-'+str(year_time[i])), '%m-%Y'))
            else:
                # correcting for the last element
                if month_time[i] == month_time[i+1]:
                    ret_tt_time.append(datetime.strptime((str(month_time[i])+'-'+str(year_time[i])), '%m-%Y'))
                else:
                    ret_tt_time.append(datetime.strptime((str(month_time[i])+'-'+str(year_time[i])), '%m-%Y'))
                    ret_tt_time.append(datetime.strptime((str(month_time[i+1])+'-'+str(year_time[i+1])), '%m-%Y'))
    return ret_tt_time

def make_datetime(raw_time):
    try:
        tt = [datetime.strptime(x,'%d-%b-%y') for x in raw_time]
    except:
        tt = [datetime.strptime(x,'%d-%m-%y') for x in raw_time]
    return tt

def make_float(y):
    try:
        tt = float(str(y).replace(',','').replace('$','').replace('(','').replace(')',''))
    except:
        tt = 0
    return tt

def monthly_avg(time,y):
    y = [make_float(x) for x in y]
    month_time = [x.month for x in time]
    year_time = [x.year for x in time]
    tt = []
    ret_tt_y = []
    for i in range(len(y)-1):
        if month_time[i] == month_time[i+1]:
            tt.append(y[i])
        else:
            if i != len(y)-2:
                tt.append(y[i])
                ret_tt_y.append(np.average(tt))
                tt = []
            else:
                # correcting for the last element
                if month_time[i] == month_time[i+1]:
                    tt.append(y[i])
                    tt.append(y[i+1])
                    ret_tt_y.append(np.average(tt))
                else:
                    tt.append(y[i])
                    ret_tt_y.append(np.average(tt))
                    ret_tt_y.append(y[i+1])
    return ret_tt_y

def plot_raw(y,time):
    n=60
    plt.subplot(2,1,1)
    plt.plot(time,y,'.-')
    plt.plot(time,pd.rolling_mean(y,n),'.-',color='purple')
    plt.minorticks_on()
    plt.legend(['raw','MA 60'])
    plt.grid(b=True, which='both')

    plt.subplot(2,1,2)
    plt.plot(time,np.log(y),'.-',color='black')
    plt.plot(time,pd.rolling_mean(np.log(y),n),'.-',color='green')
    plt.minorticks_on()
    plt.legend(['log','MA 60'])
    plt.grid(b=True, which='both')
    plt.show()

def plot_monthly(y_avg,time_avg):
    n=6
    plt.subplot(2,1,1)
    plt.plot(time_avg,y_avg,'.-')
    plt.plot(time_avg,pd.rolling_mean(y_avg,n),'.-',color='purple')
    plt.minorticks_on()
    plt.legend(['raw monthly','MA 6'])
    plt.grid(b=True, which='both')

    plt.subplot(2,1,2)
    plt.plot(time_avg,np.log(y_avg),'.-',color='black')
    plt.plot(time_avg,pd.rolling_mean(np.log(y_avg),n),'.-',color='green')
    plt.minorticks_on()
    plt.legend(['log monthly','MA 6'])
    plt.grid(b=True, which='both')
    plt.show()

    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/Strategic Procurement/data/Raw Materials/extracted')
all_files = os.listdir()
##all_files = [all_files[-1]]
for file_name in all_files:
    if file_name != 'monthly':
        tt = []
        file = pd.read_csv(file_name)
        print(file_name,file.shape)
        time = make_datetime(file.iloc[:,0])
        tt.append(agg_time(time))
        for i in range(1,file.shape[1]):
            tt.append(monthly_avg(time,file.iloc[:,i]))
        tt = pd.DataFrame(tt).transpose()
        tt.columns = file.columns
        tt.to_csv(os.getcwd()+'/monthly/'+file_name.split('.')[0]+'_monthly.csv',index=False)

