import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
df =  df[["52w H","High","365 d % chng","% Chng"]]


def pearsonCorr(x,y):
    N = len(x)
    num = N*(x*y).sum() - (x.sum()*y.sum())
    denom =  np.sqrt( (N*((x**2).sum())-x.sum()**2) * (N*((y**2).sum())-(y.sum()**2)) )
    R = num/denom
    return R

def slope(x,y):
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m_num = ((x-x_mean)*(y-y_mean)).sum()
    m_dem = ((x-x_mean)**2).sum()
    m = m_num/m_dem
    intercept = (y_mean-(x_mean)*m)
    return m,intercept


testDf = pd.DataFrame(zip([3,5,4,6,2],[3,4,5,2,6]))
m,intercept = slope(df["% Chng"],df["365 d % chng"])
plt.scatter(df["% Chng"],df["365 d % chng"])
plt.plot(df["% Chng"],intercept+m*df["% Chng"])
plt.scatter(df["% Chng"].mean(),df["365 d % chng"].mean())
plt.show()




