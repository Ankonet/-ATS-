import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OLB_method import OLB_method
from get_breakpoint_series import get_breakpoint_series
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False #让图片显示中文

'''
一、导入所需的股票数据
'''
os.chdir(r'上证50股票数据20180831-20210831')
filelist = os.listdir() #获取文件名列表
stock_name = [] #股票名字列表
dflist = []
for file_name in filelist:
    df = pd.read_excel(file_name, nrows=728)
    stock_name.append(df.iloc[1, 1]) #读取股票的名字
    df = df.iloc[:,[7,8,13,20,21,22,23]] #取出特征所在列：收盘价、成交量、换手率、市盈率、市净率、市销率、市现率
    df = pd.DataFrame(df,dtype=float)
    dflist.append(df)
data = dflist[0]
for i in range(1,len(dflist)):
    data = pd.concat([data,dflist[i]],axis=1)
data_columns1 = np.array([[name]*7 for name in stock_name]).ravel()
data_columns2 = list(dflist[0].columns)*50
data.columns = [data_columns1,data_columns2]
#data即为导入的一手股票数据

'''
二、数据预处理（缺失值填充+标准化）
'''
#将缺失较为严重的三只股票剔除
data.drop(['闻泰科技','中泰证券','中金公司'],axis=1,inplace=True)
data_st = StandardScaler().fit_transform(data) #标准化
data_st = pd.DataFrame(data_st).interpolate(method='linear') #线性插值填充
data_st.columns = data.columns #恢复列名与索引

'''
三、断点序列计算
'''
for i in range(data_st.shape[1]):
    data_st.iloc[:,i] = get_breakpoint_series(data_st.iloc[:,i], h=20, filled=True)

'''
三*：取出收盘价序列
'''
for i in ['闻泰科技','中泰证券','中金公司']:
    stock_name.remove(i) #将股票名中移除掉该三只股票
data_st_shoupan = pd.DataFrame()
for stock in stock_name:
    data_st_shoupan = pd.concat([data_st_shoupan, data_st[stock]['收盘价(元)']], axis=1)
data_st_shoupan.columns = stock_name
#进行断点查找
for i in range(data_st_shoupan.shape[1]):
    data_st_shoupan.iloc[:,i] = get_breakpoint_series(data_st_shoupan.iloc[:,i], h=20, filled=True)

'''
四、进行OLB聚类分析（time warning）
'''
OLB_method(h=20, class_name=stock_name, data=data_st, n=7, k=728-20+1)
OLB_method(h=20, class_name=stock_name, data=data_st_shoupan, n=7, k=728-20+1)

#仅考虑收盘价，以中国石化、上海机场、浦发银行三只股票为基准的聚类结果。
#第二行为距离最近的股票，第三行为距离最远的股票
fig,axes = plt.subplots(3,3,figsize=(20,10))
name1 = ['中国石化','上海机场','浦发银行']
name2 = ['中国石油','中国平安','农业银行']
name3 = ['万华化学','恒生电子','华泰证券']
name = np.array([name1,name2,name3]).ravel()
for ax,name in zip(axes.flat,name):
    ax.plot(data[name]['收盘价(元)'],label='收盘价(元)',color='orange')
    ax.set_title(name)
    ax.legend()
fig.tight_layout()

#考虑全部指标，以中国石化、上海机场、浦发银行三只股票为基准的聚类结果。
#第二行为距离最近的股票，第三行为距离最远的股票
fig,axes = plt.subplots(3,3,figsize=(20,10))
name1 = ['中国石化','上海机场','浦发银行']
name2 = ['中国石油','华泰证券','光大银行']
name3 = ['海天味业','工商银行','海螺水泥']
name = np.array([name1,name2,name3]).ravel()
for ax,name in zip(axes.flat,name):
    ax.plot(data[name]['收盘价(元)'],label='收盘价(元)')
    ax.set_title(name)
    ax.legend()
fig.tight_layout()
plt.show()