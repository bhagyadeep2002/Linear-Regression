import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Dataset kaggle link : https://www.kaggle.com/spscientist/students-performance-in-exams?select=StudentsPerformance.csv

df = pd.read_csv('dataset.csv')
df = df[["reading score","writing score"]]


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


rel = pearsonCorr(df["reading score"],df["writing score"])
print("Pearson coefficient: ",rel)
print("Quality: ",rel**2)
testDf = pd.DataFrame(zip([3,5,4,6,2],[3,4,5,2,6])) #ignore, was just testing
m,intercept = slope(df["reading score"],df["writing score"])
plt.scatter(df["reading score"],df["writing score"])
plt.plot(df["reading score"],intercept+m*df["reading score"])
plt.scatter(df["reading score"].mean(),df["writing score"].mean())
plt.show()



