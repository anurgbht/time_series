import os
import pandas as pd
import numpy as np
from datetime import datetime
import random
import pylab as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller
import arch.unitroot as auni
from scipy.fftpack import fft,ifft
import operator
import math
from pyemd import emd
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def autocorr_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the subseries means
    sum_product = np.sum((y1-np.mean(y1))*(y2-np.mean(y2)))
    # Normalize with the subseries stds
    return sum_product / ((len(x) - lag) * np.std(y1) * np.std(y2))

def acf_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x to calculate Cov
    sum_product = np.sum((y1-np.mean(x))*(y2-np.mean(x)))
    # Normalize with var of whole series
    return sum_product / ((len(x) - lag) * np.var(x))

def monthly_avg(time,y):
    month_time = [x.month for x in time]
    year_time = [x.year for x in time]
    tt = []
    ret_tt_y = []
    ret_tt_time = []
    for i in range(len(y)-1):
        if month_time[i] == month_time[i+1]:
            tt.append(y[i])
        else:
            ret_tt_y.append(np.average(tt))
            ret_tt_time.append(datetime.strptime((str(month_time[i])+'-'+str(year_time[i])), '%m-%Y'))
            tt = []
    return pd.Series(ret_tt_y),pd.Series(ret_tt_time)

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

def differencing(y,n):
    if n > 0:
        tt = []
        for i in range(len(y)-n):
            tt.append(y[i+1]-y[i])
        return tt
    else:
        return y

def plot_ACF(y_avg,n):
    n_lags = len(y_avg)//2
    y_avg = differencing(y_avg,n)
    results_acf = acf(y_avg, nlags=n_lags-1)
    plt.subplot(2,1,1)
    plt.plot(range(n_lags),results_acf,'-*')
    plt.minorticks_on()
    plt.xlabel("lag")
    plt.axhline(y=0,c='green',ls='--')
    plt.axhline(y=0.2,c='orange',ls='--')
    plt.axhline(y=-0.2,c='orange',ls='--')
    plt.grid(True,which='both')
    plt.ylabel("value of ACF")
    plt.title("ACF on Fourier residual")

    results_pacf = pacf(y_avg, nlags=n_lags-1,method = 'ols')
    plt.subplot(2,1,2)
    plt.plot(range(n_lags),results_pacf,'-*')
    plt.minorticks_on()
    plt.xlabel("lag")
    plt.axhline(y=0,c='green',ls='--')
    plt.axhline(y=0.2,c='orange',ls='--')
    plt.axhline(y=-0.2,c='orange',ls='--')
    plt.grid(True,which='both')
    plt.ylabel("value of PACF")
    plt.title("PACF on Fourier residual")

    plt.suptitle('ACF and PACF plots with differencing = {0}'.format(n))
    plt.show()
    
def stationarity_tests(y_avg):
##    print('Results of Dickey-Fuller Test:')
##    dftest = adfuller(y_avg, autolag='AIC')
##    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
##    for key,value in dftest[4].items():
##        dfoutput['Critical Value (%s)'%key] = value
##    print(dfoutput)

    ADF_arch = auni.ADF(y_avg)
    PP_arch = auni.PhillipsPerron(y_avg)
    KPSS_arch = auni.KPSS(y_avg)
    VR_arch = auni.VarianceRatio(y_avg)
    print(ADF_arch.summary)
    print(PP_arch.summary)
    print(KPSS_arch.summary)
    print(VR_arch.summary)

def FFT(y_avg,time,nf,nd):
    y_avg = list(y_avg)
    y_avg.reverse()
    mean_val = np.average(y_avg)
    y_avg = y_avg-mean_val
    # Number of samplepoints
    N = len(y_avg)
    # sample spacing
    x = np.linspace(0.0,len(y_avg)-1, len(y_avg))
    yf = fft(y_avg)
    xf = list(np.linspace(0.0, 1/2, N//2))
    xt = [1/x for x in xf]
    
    plt.subplot(3,1,3)
    plt.bar(xt, 2/N * np.abs(yf[:N//2]))
    ret_tt = [xf,[x for x in 2/N * np.abs(yf[:N//2])]]
    plt.ylabel('Relative Amplitude')
    plt.xlabel('Time Period (months)')
    plt.minorticks_on()
    plt.grid(1,which='both')
    
    ret_yf = [np.abs(x) for x in yf]

    q = reconstruct_FFT(ret_yf[:N//2],ret_yf[:N//2],yf,xf[:N//2],xt[:N//2],nf)
##    stationarity_tests(y_avg-q)

    plt.subplot(3,1,1)
    plt.plot(time,y_avg+mean_val,time,q+mean_val)
    plt.ylabel('Coking coal index')
    plt.legend(['Raw data','Fitted'])
    plt.minorticks_on()
    plt.grid(1,which='both')

    plt.subplot(3,1,2)
    plt.plot(time,y_avg-q)
    plt.ylabel('Residual')
    residual = [x for x in y_avg-q]
    ret_tt.append(residual)
    ret_tt.append([x for x in y_avg+mean_val])
    plt.legend(['Residual'])
    plt.minorticks_on()
    plt.grid(1,which='both')
    plt.suptitle('Fourier analysis, removing cylicality')

    plt.show()

    for jj in range(nd+1):
        plot_ACF(residual,jj)
    
    return ret_tt

def reconstruct_FFT(yf,yf_backup,yf_original,xf,xt,nf):
    yf.sort()
    yf.reverse()
    tt = yf[:nf]

    for i in range(len(yf_original)):
        if np.abs(yf_original[i]) not in  tt:
            yf_original[i] = (0)
    q = [x.real for x in ifft(yf_original)]
    return q

def make_data(N):
    ## sample spacing
    T = 1/8000
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0*np.pi*x)+ 2*np.sin(120.0 * 2.0*np.pi*x) * 10*np.sin(190 * 2.0*np.pi*(x+(np.pi/7)))
    ##y = [x+random.uniform(0,2) for x in y]
    ##+ 2*np.sin(120.0 * 2.0*np.pi*x) + 3*np.cos(160.0 * 2.0*np.pi*x+(np.pi/2))
    return y

def EMD(y_avg):
    IMFs = emd(y_avg)
    return IMFs
