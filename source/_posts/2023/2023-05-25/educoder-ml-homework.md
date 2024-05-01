---
title: 头歌 - 机器学习 - Adaboost
layout: post
tags:
  - Educoder
  - ML
categories:
  - Educoder
  - ML
lang: zh-CN
abbrlink: 51008
date: 2023-05-25 22:11:56
---

# 【educoder】 机器学习 --- Adaboost

## 第1关：Boosting

### 任务描述

本关任务：根据本节课所学知识完成本关所设置的选择题。

### 相关知识

为了完成本关任务，你需要掌握：1.什么是集成学习，2.Boosting。

#### 什么是集成学习

集成学习方法是一种常用的机器学习方法，分为 bagging 与 boosting 两种方法，应用十分广泛。集成学习基本思想是：对于一个复杂的学习任务，我们首先构造多个简单的学习模型，然后再把这些简单模型组合成一个高效的学习模型。实际上，就是**“三个臭皮匠顶个诸葛亮”**的道理。

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog294198) 

集成学习采取投票的方式来综合多个简单模型的结果，按 bagging 投票思想，如下面例子：

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog294208) 

假设一共训练了 5 个简单模型，每个模型对分类结果预测如上图，则最终预测结果为： A:2  B:3  3>2  结果为 B

不过在有的时候，每个模型对分类结果的确定性不一样，即有的对分类结果非常肯定，有的不是很肯定,说明每个模型投的一票应该是有相应的权重来衡量这一票的重要性。就像在歌手比赛中，每个观众投的票记 1 分，而专家投票记 10 分。按 boosting 投票思想，如下例：

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog/294224) 

A：`(0.9+0.4+0.3+0.8+0.2)/5=0.52` B：`(0.1+0.6+0.7+0.2+0.8)/5=0.48` `0.52>0.48` 结果为 A

#### Boosting

**提升方法**基于这样一种思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合所得出的判断，要比其中任何一个专家单独的判断好。

历史上， Kearns 和 Valiant 首先提出了**强可学习**和**弱可学习**的概念。指出：在 PAC 学习的框架中，一个概念，如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么就称这个概念是强可学习的；一个概念，如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么就称这个概念是弱可学习的。非常有趣的是 Schapire 后来证明强可学习与弱可学习是等价的，也就是说，在 PAC 学习的框架下，一个概念是强可学习的充分必要条件是这个概念是弱可学习的。

这样一来，问题便成为，在学习中，如果已经发现了**弱学习算法**，那么能否将它**提升**为**强学习算法**。大家知道，发现弱学习算法通常要比发现强学习算法容易得多。那么如何具体实施提升，便成为开发提升方法时所要解决的问题。

与 bagging 不同， boosting 采用的是一个串行训练的方法。首先，它训练出一个**弱分类器**，然后在此基础上，再训练出一个稍好点的**弱分类器**，以此类推，不断的训练出多个弱分类器，最终再将这些分类器相结合，这就是 boosting 的基本思想，流程如下图：

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog294254) 

可以看出，子模型之间存在强依赖关系，必须串行生成。 boosting 是利用不同模型的相加，构成一个更好的模型，求取模型一般都采用序列化方法，后面的模型依据前面的模型。

### 编程要求

根据所学完成右侧选择题。

### 测试说明

略

### 参考答案

- 现在有一份数据，你随机的将数据分成了`n`份，然后同时训练`n`个子模型，再将模型最后相结合得到一个强学习器，这属于`boosting`方法吗？

  A、是B、不是C、不确定

- 2、对于一个二分类问题，假如现在训练了`500`个子模型，每个模型权重大小一样。若每个子模型正确率为`51%`，则整体正确率为多少？若把每个子模型正确率提升到`60%`，则整体正确率为多少？

  A、51%,60%B、60%,90%C、65.7%,99.99%D、65.7%,90%

> 参考答案：
>
> 1.B 2.C



## 第2关：Adaboost算法

### 任务描述

本关任务：用 Python 实现 Adaboost，并通过鸢尾花数据集中鸢尾花的 2 种属性与种类对 Adaboost 模型进行训练。我们会调用你训练好的 Adaboost 模型，来对未知的鸢尾花进行分类。

### 相关知识

为了完成本关任务，你需要掌握：1. Adaboost 算法原理，2. Adaboost 算法流程。

#### 数据集介绍

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog286256) 

数据集为鸢尾花数据，一共有 150 个样本，每个样本有 4 个特征，由于 Adaboost 是一个串行的迭代二分类算法，运算成本较大，为了减轻运算成本，我们只利用其中两个特征与两种类别构造与训练模型，且 adaboost 算法返回的值为 1 与 -1，所以要将标签为 0 的数据改为 -1 部分数据如下图：

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog294166)

 

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog294167)

 

数据获取代码：

```python
#获取并处理鸢尾花数据
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    #将标签为0的数据标签改为-1
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]
```

#### Adaboost算法原理

对提升方法来说，有两个问题需要回答：**一是在每一轮如何改变训练数据的权值或概率分布；二是如何将弱分类器组合成一个强分类器。**关于第 1 个问题，AdaBoost的做法是，**提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值**。这样一来，那些没有得到正确分类的数据，由于其权值的加大而受到后一轮的弱分类器的更大关注。于是，分类问题被一系列的弱分类器“分而治之”。至于第 2 个问题，即弱分类器的组合，AdaBoost采取**加权多数表决的方法，加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用，减小分类误差率大的弱分类器的权值，使其在表决中起较小的作用**。

#### Adaboost算法流程

 AdaBoost 是 AdaptiveBoost 的缩写，表明该算法是具有适应性的提升算法。

算法的步骤如下：

1.给每个训练样本(*x*1,*x*2,..,*x**N*)分配权重，初始权重*w*1均为1/*N*；

2.针对带有权值的样本进行训练，得到模型*G**m*（初始模型为*G*1）；

3.计算模型*G**m*的误分率：
$$
e_m=\sum^{N}_{i}\omega_iI(y_i\neq G_M(X_i))
$$
其中：
$$
I(y_I\neq G_M(X_i))
$$
为指示函数，表示括号内成立时函数值为 1，否则为 0。

4.计算模型$G_M$的系数：
$$
\alpha_m=\frac{1}{2}\log(\frac{1-e_m}{e_m})
$$
5.根据误分率*e*和当前权重向量$\omega_m$更新权重向量：
$$
\omega_{m+1,i}=\frac{\omega_m}{z_m}e^{-\alpha_my_iG_m(x_i)}
$$
其中$Z_m$为规范化因子：
$$
z_m=\sum_{i=1}^{m}\omega_{mi}e^{-\alpha_my_iG_m(x_i)}
$$
6.计算组合模型$f(x)=\sum_{m=1}^{M}\alpha_my_iG_m(x_i)$的误分率；

7.当组合模型的误分率或迭代次数低于一定阈值，停止迭代；否则，回到步骤 2。

### 编程要求

根据提示，在右侧编辑器的 begin-end 间补充 Python 代码，实现 Adaboost 算法，并利用训练好的模型对鸢尾花数据进行分类。

### 测试说明

只需返回分类结果即可，程序内部会检测您的代码，预测正确率高于 95% 视为过关。

### 参考答案

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# adaboost算法
class AdaBoost:
    """
        input:n_estimators(int):迭代轮数
              learning_rate(float):弱分类器权重缩减系数
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = [1.0 / self.M] * self.M
        # G(x)系数 alpha
        self.alpha = []

    # ********* Begin *********#
    def _G(self, features, labels, weights):
        """
            input:features(ndarray):数据特征
                  labels(ndarray):数据标签
                  weights(ndarray):样本权重系数
        """
        e = 0
        for i in range(weights.shape[0]):
            if (labels[i] == self.G(self.X[i], self.clif_sets, self.alpha)):
                e += weights[i]
        return e

    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self, weights, a, clf):
        return np.sum(weights * np.exp(-a * self.Y * self.G(self.X, clf, self.alpha)))

    # 权值更新
    def _w(self, a, clf, Z):
        w = np.zeros(self.weights.shape)
        for i in range(self.M):
            w[i] = weights[i] * np.exp(-a * self.Y[i] * G(x, clf, self.alpha)) / Z
        self.weights = w

    # G(x)的线性组合
    def G(self, x, v, direct):
        result = 0
        x = x.reshape(1, -1)
        for i in range(len(v)):
            result += v[i].predict(x) * direct[i]
        return result

    def fit(self, X, y):
        """
            X(ndarray):训练数据
            y(ndarray):训练标签
        """
        # 计算G(x)系数a
        self.init_args(X, y)

    def predict(self, data):
        """
            input:data(ndarray):单个样本
            output:预测为正样本返回+1，负样本返回-1
        """
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
        ada.fit(self.X, self.Y)
        data = data.reshape(1, -1)
        predict = ada.predict(data)
        return predict[0]
    # ********* End *********#

```



## 第3关：sklearn中的Adaboost

### 任务描述

本关任务：你需要调用 sklearn 中的 Adaboost 模型，并通过癌细胞数据集对 Adaboost 模型进行训练。我们会调用你训练好的 Adaboost 模型，来对未知的癌细胞进行识别。

### 相关知识

为了完成本关任务，你需要掌握：1. AdaBoostClassifier。

#### 数据集介绍

乳腺癌数据集，其实例数量是 569 ，实例中包括诊断类和属性，帮助预测的属性一共 30 个，各属性包括为 radius  半径（从中心到边缘上点的距离的平均值），texture  纹理（灰度值的标准偏差）等等，类包括： WDBC-Malignant  恶性和  WDBC-Benign  良性。用数据集的 80% 作为训练集，数据集的 20% 作为测试集，训练集和测试集中都包括特征和诊断类。

想要使用该数据集可以使用如下代码：

```python
from sklearn.datasets import load_breast_cancer
#加载数据
cancer = load_breast_cancer()
#获取特征与标签
x,y = cancer['data'],cancer['target']
#划分训练集与测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=666)
```

数据集中部分数据与标签如下图所示：

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog/312821)

 

![img](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog/312823)

 

#### AdaBoostClassifier

 AdaBoostClassifier 的构造函数中有四个常用的参数可以设置：

-  algorithm ：这个参数只有 AdaBoostClassifier 有。主要原因是scikit-learn 实现了两种 Adaboost 分类算法， SAMME 和 SAMME.R。两者的主要区别是弱学习器权重的度量， SAMME.R 使用了概率度量的连续值，迭代一般比 SAMME 快，因此 AdaBoostClassifier 的默认算法 algorithm 的值也是 SAMME.R；
-  n_estimators ：弱学习器的最大迭代次数。一般来说 n_estimators 太小，容易欠拟合，n_estimators 太大，又容易过拟合，一般选择一个适中的数值。默认是 50；
-  learning_rate ：AdaBoostClassifier 和 AdaBoostRegressor 都有，即每个弱学习器的权重缩减系数 ν，默认为 1.0；
-  base_estimator ：弱分类学习器或者弱回归学习器。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的一般是 CART 决策树或者神经网络 MLP。

和 sklearn 中其他分类器一样，AdaBoostClassifier 类中的 fit 函数用于训练模型，fit 函数有两个向量输入：

-  X ：大小为**[样本数量,特征数量]**的 ndarray，存放训练样本；
-  Y ：值为整型，大小为**[样本数量]**的 ndarray，存放训练样本的分类标签。

AdaBoostClassifier 类中的 predict 函数用于预测，返回预测标签， predict 函数有一个向量输入：

 X ：大小为**[样本数量,特征数量]**的 ndarray，存放预测样本 AdaBoostClassifier 的使用代码如下：

```python
ada=AdaBoostClassifier(n_estimators=5,learning_rate=1.0)
ada.fit(train_data,train_label)
predict = ada.predict(test_data)
```

### 编程要求

在 begin-end 区域内填写`ada_classifier(train_data,train_label,test_data)`函数完成癌细胞识别任务，其中：

- train_data：训练样本；
- train_label：训练标签；
- test_data：测试样本。

### 测试说明

只需返回预测结果即可，程序内部会检测您的代码，预测正确率高于 95% 视为过关。

### 参考答案

```python
# encoding=utf8
from sklearn.ensemble import AdaBoostClassifier


def ada_classifier(train_data, train_label, test_data):
    """
        input:train_data(ndarray):训练数据
              train_label(ndarray):训练标签
              test_data(ndarray):测试标签
        output:predict(ndarray):预测结果
    """
    # ********* Begin *********#
    ada = AdaBoostClassifier(n_estimators=80, learning_rate=1.0)
    ada.fit(train_data, train_label)
    predict = ada.predict(test_data)
    # ********* End *********#
    return predict

```
 