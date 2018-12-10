Classification  Schoolwork

@author：刘士坤 2016011371 

9个类别的高光谱数据分类 ，训练、验证模型。

## 复习相关知识

### 对于python处理.mat文件的方法

| scipy.io | 数据输入输出 |
| -------- | ------------ |
| numpy    | 数据处理     |

matlab从图片处理得到的label信息都会以.mat文件供python读取，同时也python产生的结果信息也需要matlab来做进一步的处理 。

matlab和python间的数据传输一般是基于matlab的文件格式.mat，python中numpy和scipy提供了一些函数，可以很好的对.mat文件的数据进行读写和处理。  

在这里numpy作用是提供Array功能映射matlab里面的Matrix，而scipy提供了两个函数loadmat和savemat来读写.mat文件。 

参考：[scipy包含致力于科学计算中常见问题的各个工具箱](https://www.jianshu.com/p/1a3db06e786d)

​	    [NumPy IO](http://www.runoob.com/numpy/numpy-io.html)



## 代码的书写

### 导入相应的库 

### 1.将高光谱数据train导入，并按照8 : 2的比例划分成训练集与验证集

#### 1）.mat文件处理

######使用字典存储所有数据
```
data_dict = {}
data_path = "./data/train"
for i in os.listdir(data_path):
    m = sio.loadmat(os.path.join(data_path,i))
    data_dict[i.split(".")[0]] = m[i.split(".")[0]]
```

######data存储
```
data = []
for i in data_dict.keys():
    for j in data_dict[i]:
        data.append(j)
data = np.array(data)
data
```

###### label存储

```
label=[]
for i in data_dict.keys():
    y = re.sub("\D","",i)
    for j in range(len(data_dict[i])):
        label.append(y)
label = np.array(label,dtype=np.int64)
```

#### 2)按照8 : 2的比例划分成训练集与验证集 

```
X_train,X_valid,y_train,y_valid = train_test_split(data,label,test_size=0.2,shuffle=True)
```

### 2.构建分类器模型

数据预处理

```
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
```

svc分类器 

```
svc = SVC(C=19,gamma=0.01,class_weight='balanced')
svc.fit(X_train,y_train)
```

```
y_pred = svc.predict(X_valid)
print(accuracy_score(y_valid,y_pred))
```

### 3.测试集

将高光谱数据测试集test导入

```
test_data = np.array(sio.loadmat("./data/test/data_test_final.mat")['data_test_final'], dtype=np.float64)
```

```
x_test = scaler.transform(test_data)
```

```
y_test = svc.predict(x_test)
```

导出标签数据

```
data = pd.DataFrame(y_test)
data.to_csv("./data/test.csv")
```

## 作业完成概括

复习相关知识后，进行作业的完成。

使用SVM实现9个类别的高光谱数据分类 ，训练、验证模型。

