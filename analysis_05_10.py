import os
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pylab as plt
import time_series_module_15_09 as tm

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/Strategic Procurement/data/Raw Materials/extracted/')

dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%b-%y')
dat = pd.read_csv('indices.csv', parse_dates=['date'],date_parser=dateparse)
time = dat.iloc[:,0]
y = dat.iloc[:,1]
time = time.loc[~y.isnull()]
y = y.loc[~y.isnull()]
time = list(time)
y = list(y)
y.reverse()

y_avg,time_avg = tm.monthly_avg(time,y)

####tm.plot_raw(y,time)
####tm.plot_monthly(y_avg,time_avg)
####tm.plot_ACF(y_avg)
####tm.stationarity_tests(y_avg)

ret_tt=tm.FFT(y_avg,time_avg,12,1)
ret_tt = pd.DataFrame(ret_tt)
ret_tt = ret_tt.transpose()
ret_tt.to_csv('output.csv')
##IMFs = tm.EMD(y_avg)
