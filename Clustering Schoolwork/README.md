Clustering Schoolwork

@author：刘士坤 2016011371

以谱聚类对鸢尾花数据集进行处理，得到类似如下图所示（Normalized Cut），并输出正确率。

## 复习相关知识

### 对于谱聚类的理解

谱聚类的目的是使得簇内相似度高，簇间相似度低，也就是要使得被切割的边的权重尽可能的低， 但一个簇与簇外的顶点之间的边除了与簇聚的好 坏有关之外，还与簇的大小等因素有关，所以单纯的以来表示一个簇的质量并不合适。常用的聚类目标函数有最小化目标和最大化目标两种，其中最小化目标最常用的有比例割和归一割。

聚类有两个要求：使得簇内相似度尽可能的高，簇间 相似度尽可能的低，图的聚类中对目标函数的优化就是朝着这两方面努力。比例割和归一割是使得簇间相似度尽可能的低，所以这两种目标函数需要最小化，而平均割与模块度表示的是簇内的相似度，所以需要最大化。

谱聚类算法流程 

1. 如果数据集是向量的形式，首先计算其相似度矩阵作为邻接矩阵A，如果是图的形式，可以直接得到A。
2. 由A得到度数矩阵和拉普拉斯矩阵。
3. 如果选择比例割，求拉普拉斯矩阵的特 征值和特征向量，如果选择归一割，则求归一化后的对称或者不对称拉普拉斯矩阵的特征值和特征向量。
4. 找到最小的k个特征值对应的特征向量组成矩阵U。
5. 对U的每一行进行归一化，得到矩阵Y。 
6. 将Y的每一行看成是一个数据，以kmeans算法或其他快速聚类算法进行聚类。

### 画图的学习

绘制基本网络图
基本流程：导入networkx，matplotlib包， 建立网络，
绘制网络 nx.draw()，建立布局 pos ，添加点线

## 代码的书写

### 1.将鸢尾花数据集画成图的形式

```python
#绘制网络 nx.draw()
plt.figure(figsize=(10,10))
G = nx.Graph()
#根据带权邻接矩阵画图，由近邻K（n_k）决定是否相连
A_mat=adj_mat(data)
for i in range(len(A_mat)):
    for j in range(len(A_mat)):
        if(i in n_k[j] and j in n_k[i]):
            G.add_edge(i,j,weight=A_mat[i,j])

# 由生成的节点将网络图上的边按权重分成3组
edge1=[]
edge2=[]
edge3=[]
for (m,n,l) in G.edges(data='weight'):
    if l >= 0.96:
        edge1.append((m,n))
    elif l < 0.92:
        edge2.append((m,n))
    else:
        edge3.append((m,n))
        
#根据pca降维确定的点位置
pos= reduced_X
# 画节点  
nx.draw_networkx_nodes(G, pos,node_color='r',node_shape='o',node_size=40)
# 画边
nx.draw_networkx_edges(G, pos, edgelist=edge1,alpha=0.5,edge_color='k',style='solid')
nx.draw_networkx_edges(G, pos, edgelist=edge2,alpha=0.3,edge_color='b',style='solid')
nx.draw_networkx_edges(G, pos, edgelist=edge3,alpha=0.2,edge_color='g',style='solid')

plt.savefig("鸢尾花数据集画成图.png")# 保存图片用于之后的作业提交
plt.show()
```

### 2.确定一个合适的**阈值**，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边

```python
# 计算每个样本的K近邻
def N_k(A,K):
    N = []
    for i in range(len(A)):
        j = np.argsort(A[i])
        j = j[-k:-1]
        N.append(j)
    return np.array(N)
```

### 3.求取带权邻接矩阵

```python
#计算带权邻接矩阵 A
def adj_mat(data):
    N = data.shape[0]
    A = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i+1):
            A[i,j] = np.dot(data[i]-data[j],data[i]-data[j])
            if(i!=j):
                A[j,i] = A[i,j] 
                
    A_mat = np.exp(-0.5*A)
    return  A_mat
```
### 4.根据邻接矩阵进行聚类

```python
#拉普拉斯矩阵
def Laplace_matrix(X,k):
    #相似度矩阵作为邻接矩阵A
    A = adj_mat(X)             #开始实现的带权邻接矩阵
    #由A得到度数矩阵和拉普拉斯矩阵
    D = np.diag(A.sum(axis=0)) #度矩阵D
    L = D - A                  # 拉普拉斯矩阵
    D = np.linalg.inv(np.sqrt(D))
    L = D.dot(L).dot(D)
    return L

#取前K个单位化的特征向量,找到最小的k个特征值对应的特征向量组成矩阵U
def spectral_clustering(X,k):
    L=Laplace_matrix(X,k)
    m, n = np.linalg.eig(L)
    i = np.argsort(m)[:k]
    Vectors = n[:, i]
    Vectors=normalization(Vectors)
    return Vectors

#归一化处理
def normalization(Vectors):
    #对U的每一行进行归一化，得到矩阵Y
    normalizer = np.linalg.norm(Vectors, axis=1)  # normalized
    normalizer = np.repeat(np.transpose([normalizer]), k, axis=1)
    Vectors = Vectors / normalizer
    return Vectors
    
clf = KMeans(n_clusters=3)
y = clf.fit_predict(spectral_clustering(data,k))
```



### 5.将聚类结果可视化，重新转换成图的形式，其中每一个簇应该用一种形状表示，比如分别用圆圈、三角和矩阵表示各个簇

```python
#绘制网络 nx.draw()
plt.figure(figsize=(10,10))
G = nx.Graph()

# 根据带权邻接矩阵画图
A = adj_mat(data)
for i in range(len(A)):
    for j in range(len(A)):
        if(i in n_k[j] and j in n_k[i]):
            G.add_edge(i,j, weight=A[i,j])
            
#由标签将节点分成3组
node1=[i for i in range(150) if y[i] == 2]
node2=[i for i in range(150) if y[i] == 1]
node3=[i for i in range(150) if y[i] == 0]

# 由生成的节点将网络图上的边按权重分成3组
edge1=[]
edge2=[]
edge3=[]
for (m,n,l) in G.edges(data='weight'):
    if l >= 0.96:
        edge1.append((m,n))
    elif l < 0.92:
        edge2.append((m,n))
    else:
        edge3.append((m,n))
        
# 画出节点
nx.draw_networkx_nodes(G, pos, node_size=40, nodelist=node1, node_shape='o')
nx.draw_networkx_nodes(G, pos, node_size=40, nodelist=node2, node_shape='^')
nx.draw_networkx_nodes(G, pos, node_size=40, nodelist=node3, node_shape='s')
# 画出边
nx.draw_networkx_edges(G, pos, edgelist=edge1, width=1, alpha=0.2, edge_color='k', style='solid')
nx.draw_networkx_edges(G, pos, edgelist=edge2, width=1, alpha=0.6, edge_color='b', style='dashed')
nx.draw_networkx_edges(G, pos, edgelist=edge3, width=1, alpha=0.4, edge_color='g', style='solid')

plt.savefig("聚类之后的图.png")# 保存图片用于之后的作业提交
plt.show()
```

### 6.求得分簇正确

```python
#计算正确率
correct = 0
for i in range(150):
    if target[i] == y[i]:
        correct=correct+1
#输出正确率
print("Acc:","{}%".format(correct/150*100))       
```

### 7.完成代码的描述文档

1. 代码说明文档中要讲明调参的过程及原因。

   调参，对于K值的手动控制，主要是由于K-means中的质心初始化方式是随机的，聚类结果不稳定，由正确率调整。

2. 算法正确率自行根据鸢尾花数据集计算并输出，描述文档给出说明为什么可以达到这种正确率，比如正确率90%，并说明错误的10%为什么会错。

   正确率为78%，主要是聚类结果不稳定，错分的点导致，簇中心的调整。

## 作业完成概括

复习相关知识后，进行作业的完成。
