'''
该函数返回断点序列的值并可决定是否自动进行线性插值
其中q为输入序列，h为步长
j为初始出发点，r为该点处斜率方向，取值为{-1，1，0}
函数返回含有断点序列的Series数据类型
'''
import numpy as np
import pandas as pd
from pandas.core.series import Series

def get_breakpoint_series(q:Series, h:int, j=0, r=0, filled=False)-> Series:
    q = q.tolist() #将series转化为列表类型，方便计算
    x, y = [0], [q[0]] #x为时间，y为断点序列值
    m = len(q) #序列长度

    while j + h < m:
        if r == 0:
            s = (q[j + h - 1] - q[0]) / (j + h - 1)
            r = np.sign(s)
            j = j + 1
        else:
            s = (q[j + h - 1] - q[j]) / (h - 1)
            if np.sign(s) == r:
                j = j + 1
            else:
                if np.sign(s) > 0:
                    vd = min(q[j:j + h])
                    x.append(q[j:j+h].index(vd)+j)
                    y.append(vd)
                    j = q[j:j+h].index(vd)+j + 1
                    r = np.sign(s)
                else:
                    vd = max(q[j:j + h])
                    x.append(q[j:j+h].index(vd)+j)
                    y.append(vd)
                    j = q[j:j+h].index(vd)+j + 1
                    r = np.sign(s)
    x.append(len(q) - 1) #将原序列末尾序列值添加入断点序列
    y.append(q[-1])

    if filled == True: #是否要进行序列填充
        time = pd.DataFrame(range(q.shape[0]), columns=['time'])
        point = pd.DataFrame(zip(x, y), columns=['time', 'point'])
        df = pd.merge(time, point, how='left')
        df = df.interpolate()  # 对断点序列进行线性插值
        return df['point']
    else:
        return pd.Series(y,index=x)