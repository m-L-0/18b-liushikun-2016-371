TensorFlow Schoolwork

@author：刘士坤 2016011371 

使用TensorFlow 设计K近邻模型，并使用鸢尾花数据集训练、验证模型。

## 复习相关知识

### 对于KNN算法的理解

KNN算法的本质是在训练过程中，它将所有训练样本的输入和输出标签(label)都存储起来。测试过程中，计算测试样本与每个训练样本的距离，选取与测试样本距离最近的前k个训练样本。然后对着k个训练样本的label进行投票，票数最多的那一类别即为测试样本所归类。 

KNN算法是一种最简单最直观的分类算法。它的训练过程保留了所有样本的所有特征，把所有信息都记下来，没有经过处理和提取。而其它机器学习算法包括神经网络则是在训练过程中提取最重要、最有代表性的特征。在这一点上，KNN算法还非常不够“智能”。

## 代码的书写

### 1.数据下载与归类

将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集（不使用Dataset API）

Iris数据集每个样本x包含了花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）、花瓣宽度（petal width）四个特征。样本标签y共有三类，分别是Setosa，Versicolor和Virginica。Iris数据集总共包含150个样本，每个类别由50个样本，整体构成一个150行5列的二维表。

将每个类别的所有样本分成训练样本（training set）和测试样本（test set），各占所有样本的比例分别为80%，20%。进行40：10的划分后合并。

### 2.KNN训练函数和预测函数

#### 1）设计模型的思路

KNN算法的构建

KNN的训练过程实际上是一种数据标类、数据存储的过程。

```python
#构建图模型
# 输入占位符，X_train为训练集的占位符，X_test为一个验证样本的占位符
z_train = tf.placeholder("float", [None, 4])
z_test = tf.placeholder("float", [4])
#构建计算距离
distance = tf.reduce_sum(tf.abs(tf.add(z_train, tf.negative(z_test))), reduction_indices=1)

# 预测函数   TensorFlow设计K近邻模型
def knn(K): 
    with tf.Session() as sess:
        #字典存储
        pred = [] 
        #测试集循环
        for i in range(len(X_test)):
            #距离矩阵
            distance_matrix = sess.run(distance, feed_dict={z_train:X_train,z_test:X_test[i]}) 
            #根据K值，选择最可能属于的类别
            # 矩阵排序,先取前K个
            knn_fir = np.argsort(distance_matrix)[:K]
            #再三个类别投票表决 
            Iris_class=[0, 0, 0]
            for m in knn_fir:
                if(y_train[m]==0):
                    Iris_class[0] += 1
                elif(y_train[m]==1):
                    Iris_class[1] += 1
                else:
                    Iris_class[2] += 1
            y_pred = np.argmax(Iris_class)
            pred.append(y_pred)
        return pred
```

KNN的测试过程是核心部分。其中，有两点需要注意：

- 衡量距离的方式
- K值的选择

##### 衡量距离的方式

KNN距离衡量一般有两种方式：L1距离和L2距离。

##### K值的选择

KNN中K值的选择至关重要，K值太小会使模型过于复杂，造成**过拟合**；K值太大会使模型分类模糊，造成**欠拟合**。可以选择不同的K值，通过验证来决定K值大小。代码中，将K设定为可调参数。



#### 2）训练模型

由图所示，选择合适的K值



#### 3）验证模型

对测试集进行预测

选择完合适的K值之后，就可以对测试集进行预测分析了。



## 作业完成概括

复习相关知识后，进行作业的完成。

实现对鸢尾花数据集与KNN算法的熟悉，练习tensorflow的运行。

