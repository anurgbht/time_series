import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
from sklearn.decomposition import PCA

#########################################################################################################
#########################################################################################################
#########################################################################################################

def my_mape(y,y_pred):
    tt = [np.abs((t1-t2)/t1) for t1,t2 in zip(y,y_pred)]
    return 100*np.average(tt)


def normal_regression(X,y):
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X, y)

    # Make predictions using the testing set
    y_pred = regr.predict(X)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y, y_pred))
    print('MAPE : %.2f' % my_mape(y, y_pred))
    # Plot outputs
    my_plot(y,y_pred)

def PCA_regression(X,y,n):
    pca = PCA(n_components=n)
    pca.fit(X)
    X = pca.transform(X)
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X, y)

    # Make predictions using the testing set
    y_pred = regr.predict(X)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y, y_pred))
    print('MAPE : %.2f' % my_mape(y, y_pred))
    # Plot outputs
    my_plot(y,y_pred)

def my_plot(y,y_pred):
    plt.plot(list(y))
    plt.plot(list(y_pred))
    plt.legend(['actual data','predicted data'])
    plt.ylabel('Platts index')
    plt.xlabel('Time (months)')
    plt.grid(1,which='both')
    plt.minorticks_on()
    plt.suptitle('PCA Regression on external variables')
    plt.show()

def fill_blank(tt_temp):
    tt = []
    tt_temp = list(tt_temp)
    for i in range(len(tt_temp)-1):
        if np.isnan(tt_temp[i]):
            tt.append(tt_temp[i+1])
        else:
            tt.append(tt_temp[i])
    tt.append(tt_temp[i+1])
    return tt

def fill_blank_wrapper(dat):
    print('All shape :',dat.shape)
    print('Initial available : ',dat.dropna().shape)
    col_names = dat.columns
    tt = []
    tt.append(list(dat.iloc[:,0]))
    for i in range(1,dat.shape[1]):
        temp = dat.iloc[:,i]
        tt.append(fill_blank(temp))
        
    dat = pd.DataFrame(tt).transpose()
    dat.columns = col_names
    print('Final available : ',dat.dropna().shape)
    return dat

def make_lagged(X,y,n):
    y = y.iloc[n:]
    X = X.iloc[:-n,:]
    print(X.shape)
    print(y.shape)
    return X,y
    
#########################################################################################################
#########################################################################################################
#########################################################################################################
os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/Strategic Procurement/data/Raw Materials/')

dat = pd.read_csv('for_regression_reduced_12_10.csv')
for i in range(10):
    dat = fill_blank_wrapper(dat)
dat = dat.dropna()
print('Final available : ',dat.dropna().shape)
##dat.to_csv('temp_regression.csv',index=False)
X = dat.iloc[:,2:]
y = dat.iloc[:,1]
X,y = make_lagged(X,y,3)
PCA_regression(X,y,10)
