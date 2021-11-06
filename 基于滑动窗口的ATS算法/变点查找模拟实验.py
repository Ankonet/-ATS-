import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_breakpoint_series import get_breakpoint_series
'''
一、将事先模拟生成好的序列读入
'''
data = pd.read_csv('data.csv',index_col=0)

'''
二、获取断点
'''
h = 20
fig, axes = plt.subplots(3,figsize = (12,15))
for index, ax in enumerate(axes):
    q = data.iloc[:,index]
    ax.plot(q,label='original seires')
    y = get_breakpoint_series(q,h)
    ax.plot(y,label='new ATS series')
    ax.legend()
    if ax == axes[0]:
        ax.set_title('h=20')
    ax.set_xticks(range(0,310,50))
plt.show()

'''
三、现尝试改变原始窗格的尺寸，来观察断点个数及位置。依次选取窗格大小为10，20，30
'''
fig, axes = plt.subplots(3,3,figsize = (30,18))
h_lst = [10,20,30]
for row in range(3):
    q = data.iloc[:,row]
    for col,h in enumerate(h_lst):
        axes[row,col].plot(q,label='original seires')
        y = get_breakpoint_series(q,h)
        axes[row,col].plot(y,label='new ATS series')
        axes[row,col].legend()
        
        if row == 0:
            axes[row,col].set_title('h={}'.format(h_lst[col]))
plt.show()
#从图中可以看出窗格尺寸对断点的查找影响不明显

'''
四、现绘制学习曲线，观察随着窗格尺寸的增大，断点个数的变化趋势
'''
h_lst = range(5,51,1)
fig,ax = plt.subplots(figsize=(15,7))
for i in range(3):
    q = data.iloc[:,i]
    x_num_lst = []
    for h in h_lst:
        y = get_breakpoint_series(q,h)
        x_num_lst.append(y.shape[0])
    ax.plot(h_lst,x_num_lst,marker = 'o',label = 'series'+str(i+1))
    ax.set_xlabel('h-value')
    ax.set_ylabel('num of breakpoint')
    ax.legend()
plt.show()
#从图中可以看出，当h>20时，曲线逐渐趋于平缓

