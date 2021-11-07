import pandas as pd
from cal_series_dist import cal_series_dist

def OLB_method(h, class_name, data, n, k):
    '''
    用OLB方法返回距离每一条序列最近的序列，并打印报表
    h为滑窗长度
    data为全部多维序列的concat
    n为多元时间序列的维度
    k为窗格的个数，计算公式为：k = 序列长度-滑窗长度+1
    '''
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
                y_dist.append(cal_series_dist(TimeSeries_set,n))
            ans[y_name[y_dist.index(min(y_dist))]] += 1
        print('基准序列为{}'.format(x_name))
        print(dict(sorted(ans.items(),key=lambda x:x[1],reverse=True)))
        print('距离最近的序列为{}'.format(max(ans,key=ans.get)))
        print('------------------------------------------------------------------')