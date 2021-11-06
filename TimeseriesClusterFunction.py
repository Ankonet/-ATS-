import numpy as np
import pandas as pd
from pandas.core.series import Series
import scipy.cluster.hierarchy as sch



'''
该函数能够取出相关系数矩阵的上三角元素，并排列为n维向量
其中输入的matrix为array
'''
def get_corr(matrix):
    ans = []
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
             ans.append(matrix[i,j])
    return ans


'''
计算两个n维时序间的距离,其中TimeSeries_set为concat的两个n维时间序列，
feature_idx表示特征的序号且0<feature_idx<=n,
n为多元时间序列的维度。
'''
def cal_dist_each_feature(TimeSeries_set, feature_idx, n):
    

    # 取出两组序列的第一个特征
    data = TimeSeries_set.iloc[:, [i for i in range(feature_idx, TimeSeries_set.shape[1], n)]]
    # 计算相关系数
    corr = abs(data.corr().iloc[0, 1])
    data = np.array(data).T
    dist = sch.distance.pdist(data, 'euclidean')[0] / TimeSeries_set.shape[0]
    return (corr ** (-.5)) * dist

def cal_dist(TimeSeries_set, n):
    ts_dist = 0
    for feature_idx in range(n):
        ts_dist += cal_dist_each_feature(TimeSeries_set, feature_idx, n)
    return ts_dist

'''
用OLB方法返回距离每一条序列最近的序列，并打印报表
h为滑窗长度
data为全部多维序列的concat
n为多元时间序列的维度
k为窗格的个数，计算公式为：k = 序列长度-滑窗长度+1
'''
def OLB_method(h, class_name, data, n, k):
    for x_name in class_name:
        y_name = class_name[:]
        y_name.remove(x_name)
        ans = dict(zip(y_name,[0 for i in range(len(y_name))]))
        for start in range(k):
            x_ts = data[x_name].iloc[start:start+h]
            y_dist = []
            for y_ts_name in y_name:
                y_ts = data[y_ts_name].iloc[start:start+h]
                TimeSeries_set = pd.concat([x_ts,y_ts],axis = 1)
                y_dist.append(cal_dist(TimeSeries_set,n))
            ans[y_name[y_dist.index(min(y_dist))]] += 1
        print('基准序列为{}'.format(x_name))
        print(dict(sorted(ans.items(),key=lambda x:x[1],reverse=True)))
        print('距离最近的序列为{}'.format(max(ans,key=ans.get)))
        print('------------------------------------------------------------------')