TensorFlow Schoolwork

使用TensorFlow  Eager模式设计K近邻模型，并使用鸢尾花数据集训练、验证模型。

## 复习相关知识

### 对于KNN算法的理解

KNN算法的本质是在训练过程中，它将所有训练样本的输入和输出标签(label)都存储起来。测试过程中，计算测试样本与每个训练样本的距离，选取与测试样本距离最近的前k个训练样本。然后对着k个训练样本的label进行投票，票数最多的那一类别即为测试样本所归类。 

KNN算法是一种最简单最直观的分类算法。它的训练过程保留了所有样本的所有特征，把所有信息都记下来，没有经过处理和提取。而其它机器学习算法包括神经网络则是在训练过程中提取最重要、最有代表性的特征。在这一点上，KNN算法还非常不够“智能”。

### 对于eager模式的理解

TensorFlow下的eager模式是一个命令式编程模式。

不需要任何占位符或会话（sessions）。在Eager 模式下能够使用Python的debug调试工具、数据结构、控制流， 且不必再使用placeholder、session, 操作结果直接可以得到。在此种执行模式下， tensor的表现如同numpy array一般， 可以和numpy的库函数兼容。 

开启eager模式之后，TensorFlow中的大部分操作都可以使用，但操作也变得有一些不同，主要是Tensor对象的操作与流程控制部分有较大变化。引入的Eager Execution模式后, TensorFlow就拥有了类似于Pytorch一样动态图模型能力, 我们可以不必再等到see.run(*)才能看到执行结果, 可以方便在IDE随时调试代码,查看OPs执行结果。

## 代码的书写

### 1.数据下载与归类

将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集（不使用Dataset API）

Iris数据集每个样本x包含了花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）、花瓣宽度（petal width）四个特征。样本标签y共有三类，分别是Setosa，Versicolor和Virginica。Iris数据集总共包含150个样本，每个类别由50个样本，整体构成一个150行5列的二维表。

将每个类别的所有样本分成训练样本（training set）和测试样本（test set），各占所有样本的比例分别为80%，20%。进行40：10的划分后合并。

### 2.KNN训练函数和预测函数

#### 1）设计模型的思路

开启eager模式，进行KNN算法的构建

KNN的训练过程实际上是一种数据标类、数据存储的过程。首先定义一个类来实现KNN算法模块。

该类的初始化定义为：

```
class KNearestNeighbor(object):
    def __init__(self):
        pass
```

然后，在KNearestNeighbor类中定义训练函数，训练函数保存所有训练样本。

```
def train(self, X, y):
    self.X_train = X
    self.y_train = y
```

KNN的测试过程是核心部分。其中，有两点需要注意：

- 衡量距离的方式
- K值的选择

##### 衡量距离的方式

KNN距离衡量一般有两种方式：L1距离和L2距离。

一般来说，L1距离和L2距离都比较常用。需要注意的是，如果两个样本距离越大，那么使用L2会继续扩大距离，即对距离大的情况惩罚性越大。反过来说，如果两个样本距离较小，那么使用L2会缩小距离，减小惩罚。这里，使用最常用的L2距离。

##### K值的选择

KNN中K值的选择至关重要，K值太小会使模型过于复杂，造成**过拟合**；K值太大会使模型分类模糊，造成**欠拟合**。可以选择不同的K值，通过验证来决定K值大小。代码中，将K设定为可调参数。

在KNearestNeighbor类中定义预测函数：

```
def predict(self, X, k=1)
    # 计算L2距离
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))    # 初始化距离函数
    # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
    d1 = -2 * np.dot(X, self.X_train.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(X), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
    dist = np.sqrt(d1 + d2 + d3)
    # 根据K值，选择最可能属于的类别
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
        y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
        y_pred[i] = np.argmax(np.bincount(y_kclose))    # 找出k个标签中从属类别最多的作为预测类别

    return y_pred
```

#### 2）训练模型

##### 选择合适的K值

使用TensorFlow完成训练相关的代码。

首先，创建一个KnearestNeighbor实例对象。

然后，在验证集上进行k-fold交叉验证。选择不同的K值，根据验证结果，选择最佳的K值。

K值取3的时候，验证集的准确率最高。此例中，由于总体样本数据量不够多，所以验证结果并不明显。但是使用k-fold交叉验证来选择最佳K值是最常用的方法之一。

#### 3）验证模型

对测试集进行预测

选择完合适的K值之后，就可以对测试集进行预测分析了。



## 作业完成概括

复习相关知识后，进行作业的完成。

