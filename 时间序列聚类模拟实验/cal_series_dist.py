'''
计算两个n维时间序列间的距离。
其中TimeSeries_set为concat的两个n维时间序列，n为多元时间序列的维度。
'''
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import scipy.cluster.hierarchy as sch

def cal_series_dist(TimeSeries_set: DataFrame, n: int)-> float:
    ts_dist = 0
    for feature_idx in range(n):
        data = TimeSeries_set.iloc[:, [i for i in range(feature_idx, TimeSeries_set.shape[1], n)]]
        # 计算相关系数
        corr = abs(data.corr().iloc[0, 1])
        data = np.array(data).T
        dist = sch.distance.pdist(data, 'euclidean')[0] / TimeSeries_set.shape[0]
        ts_dist += (corr ** (-.5))*dist
    return ts_dist