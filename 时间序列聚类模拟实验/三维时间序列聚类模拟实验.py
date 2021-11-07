import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.kdtree import distance_matrix
from Multi_Dimensional_Data import load_data
from get_breakpoint_series import get_breakpoint_series
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from cal_series_dist import cal_series_dist

'''
一、加载数据并进行标准化
'''
data = load_data()
data_st = StandardScaler().fit_transform(data)
data_st = pd.DataFrame(data_st,columns=data.columns)
class_name = ['Aa','Ab','Ac','Ba','Bb','Bc','Ca','Cb','Cc'] #时间序列名称，其中Aa表示第A类第a条三维序列

#将生成的三维序列可视化
fig,axes = plt.subplots(3,3,figsize = (25,10))
idx = 0
for i,ax in enumerate(axes.flat):
    j = 1
    for ts_idx in range(idx,idx+3):
        ax.plot(data_st.iloc[:,ts_idx],label=data_st.columns[ts_idx]+str(j))
        ax.legend()
        j += 1
    ax.set_title(class_name[i])
    idx += 3
fig.tight_layout() #宽松排版
plt.show()

'''
二、利用最优下界法，最上述序列进行断点查找
'''
for i in range(data_st.shape[1]):
    data_st.iloc[:,i] = get_breakpoint_series(data_st.iloc[:,i], h=20, filled=True) #对标准化后的数据进行断点查找

#进行断点序列可视化
fig,axes = plt.subplots(3,3,figsize = (25,10))
idx = 0
for i,ax in enumerate(axes.flat):
    j = 1
    for ts_idx in range(idx,idx+3):
        ax.plot(data_st.iloc[:,ts_idx],label=data_st.columns[ts_idx]+str(j))
        ax.legend()
        j += 1
    ax.set_title(class_name[i])
    idx += 3
fig.tight_layout()
plt.show()

'''
三、基于断点序列，计算时序之间的距离
'''
distance_lst = []
for x in class_name:
    for y in class_name:
        if x == y:
            dist = 0
        else:
            TimeSeries_set = pd.concat([data_st[x],data_st[y]],axis = 1)
            dist = cal_series_dist(TimeSeries_set,n=3)
        distance_lst.append(dist)

distance_matrix = pd.DataFrame(np.array(distance_lst).reshape(9,9)
                               ,columns = class_name
                               ,index = class_name)

#将时序距离可视化
plt.figure(figsize=(10,10))
plt.title('the distant between series')
sns.heatmap(distance_matrix, annot=True)
plt.show()
#运用层次聚类法对距离矩阵进行聚类
model = AgglomerativeClustering(n_clusters=3, compute_distances=True, affinity="precomputed", linkage='average')
labels = model.fit_predict(distance_matrix)
print(labels) #预测标签为[2 2 2 0 0 0 1 1 1]
#聚类结果证明该法可行

#层次聚类结果可视化
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, truncate_mode='level',p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()