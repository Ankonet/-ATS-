import numpy as np
import pandas as pd
from numpy.random import RandomState
'''
该模块将模拟生成3族长度为200的三维时间序列。
故生成一共9种序列生成器，并赋予不同的随机白噪声，使之分为3组时序，每组3类*3条=9条
其中三类为A,B,C，每一类中有三个三维时间序列a,b,c
'''
def load_data():
    #确定序列的原始函数表达式
    time = np.arange(1,401,2)
    #A类序列
    series1 = np.log(time)+np.sin(time/10) #A类特征1
    series2 = np.exp(time/150) + np.cos(np.sqrt(time/30)) #A类特征2
    series3 = -np.sin(time/50) - np.log(time/20) #A类特征3
    #B类序列
    series4 = (time ** -0.5)*time #B类特征1
    series5 = 10/(1+5*np.exp(-time/40))+np.cos(time/75) #B类特征2
    series6 = np.sin(time/50) #B类特征3
    #C类序列
    series7 = np.cos(time/20) #C类特征1
    series8 = np.exp(np.sin(time/10)) #C类特征2
    series9 = -time**1.5+time**0.5 #C类特征3
    data = np.array([series1,series2,series3,series4,series5,series6,series7,series8,series9]).T
    #在数据上附加不同方差的白噪声
    data_lst = []
    num = 0
    for idx in range(0,data.shape[1],3):
        for time in range(3):
            for series in data[:,idx:idx+3].T:
                rnd = RandomState(num)
                data_lst.append(rnd.normal(0,0.6*series.std(),series.shape) + series)
                num += 5
    data_df = pd.DataFrame(data_lst).T
    data_df.columns = ['Aa','Aa','Aa','Ab','Ab','Ab','Ac','Ac','Ac',
                    'Ba','Ba','Ba','Bb','Bb','Bb','Bc','Bc','Bc',
                    'Ca','Ca','Ca','Cb','Cb','Cb','Cc','Cc','Cc']
    return data_df