---
title: 头歌 - 机器学习 - 决策树
layout: post
tags:
  - Educoder
  - ML
categories:
  - Educoder
  - ML
lang: zh-CN
mathjax: true
abbrlink: 62378
date: 2023-05-04 10:17:47
---

# 【educoder】 机器学习 --- 决策树

## 第 1 关：什么是决策树

#### 任务描述

本关任务：根据本节课所学知识完成本关所设置的选择题。

#### 相关知识

为了完成本关任务，你需要掌握决策树的相关基础知识。

##### 引例

在炎热的夏天，没有什么比冰镇后的西瓜更能令人感到心旷神怡的了。现在我要去水果店买西瓜，但什么样的西瓜能入我法眼呢？那根据我的个人习惯，在挑西瓜时可能就有这样的脑回路。

![img-1](https://data.educoder.net/api/attachments/283157)

假设现在水果店里有`3`个西瓜，它们的属性如下：

| 编号 | 瓤是否够红 | 够不够冰 | 是否便宜 | 是否有籽 |
| ---- | ---------- | -------- | -------- | -------- |
| 1    | 是         | 否       | 是       | 否       |
| 2    | 是         | 是       | 否       | 是       |
| 3    | 否         | 是       | 是       | 否       |

那么根据我的脑回路我会买`1`和`2`号西瓜。

其实我的脑回路可以看成一棵树，并且这棵树能够帮助我对买不买西瓜这件事做决策，所以它就是一棵决策树。

##### 决策树的相关概念

决策树是一种可以用于分类与回归的机器学习算法，但主要用于分类。用于分类的决策树是一种描述对实例进行分类的树形结构。决策树由结点和边组成，其中结点分为内部结点和叶子结点，内部结点表示一个特征或者属性，叶子结点表示标签（脑回路图中黄色的是内部结点，蓝色的是叶子结点）。

从代码角度来看，决策树其实可以看成是一堆`if-else`语句的集合，例如引例中的决策树完全可以看成是如下代码：

```python
if isRed:
    if isCold:
        if hasSeed:
            print("buy")
        else:
            print("don't buy")
    else:
        if isCheap:
            print("buy")
        else:
            print("don't buy")
else:
    print("don't buy")
```

因此决策树的一个非常大的优势就是模型的可理解性非常高，甚至可以用来挖掘数据中比较重要的信息。

那么如何构造出一棵好的决策树呢？其实构造决策树时会遵循一个指标，有的是按照信息增益来构建，如 ID3 算法；有的是信息增益率来构建，如 C4.5 算法；有的是按照基尼系数来构建的，如 CART 算法。但不管是使用哪种构建算法，决策树的构建过程通常都是一个递归选择最优特征，并根据特征对训练集进行分割，使得对各个子数据集有一个最好的分类的过程。

这一过程对应着对特征空间的划分，也对应着决策树的构建。一开始，构建决策树的根结点，将所有训练数据都放在根结点。选择一个最优特征，并按照这一特征将训练数据集分割成子集，使得各个子集有一个在当前条件下最好的分类。如果这些子集已经能够被基本正确分类，那么构建叶子结点，并将这些子集分到所对应的叶结点中去；如果还有子集不能被基本正确分类，那么就对这些子集选择新的最优特征，继续对其进行分割，并构建相应的结点。如此递归进行下去，直至所有训练数据子集被基本正确分类，或者没有合适的特征为止。最后每个子集都被分到叶子结点上，即都有了明确的类别。这就构建出了一棵决策树。

#### 编程要求

根据本关所学习到的知识，完成所有选择题。

#### 测试说明

平台会对你的选项进行判断，如果实际输出结果与预期结果相同，则通关；反之，则 `GameOver`。

#### 参考答案

- 1、下列说法正确的是？

  A、训练决策树的过程就是构建决策树的过程

  B、ID3 算法是根据信息增益来构建决策树

  C、C4.5 算法是根据基尼系数来构建决策树

  D、决策树模型的可理解性不高

- 2、下列说法错误的是？

  A、从树的根节点开始，根据特征的值一步一步走到叶子节点的过程是决策树做决策的过程

  B、决策树只能是一棵二叉树

  C、根节点所代表的特征是最优特征

> 1. A B
> 2. B

## 第 2 关：信息熵与信息增益

#### 任务描述

本关任务：掌握什么是信息增益，完成计算信息增益的程序设计。

#### 相关知识

为了完成本关任务，你需要掌握：

- 信息熵；
- 条件熵；
- 信息增益。

##### 信息熵

信息是个很抽象的概念。人们常常说信息很多，或者信息较少，但却很难说清楚信息到底有多少。比如一本五十万字的中文书到底有多少信息量。

直到 1948 年，香农提出了“信息熵”的概念，才解决了对信息的量化度量问题。信息熵这个词是香农从热力学中借用过来的。热力学中的热熵是表示分子状态混乱程度的物理量。香农用信息熵的概念来描述信源的不确定度。信源的不确定性越大，信息熵也越大。

从机器学习的角度来看，信息熵表示的是信息量的期望值。如果数据集中的数据需要被分成多个类别，则信息量`I(xi)`的定义如下(其中`xi`表示多个类别中的第`i`个类别，`p(xi)`数据集中类别为`xi`的数据在数据集中出现的概率表示)：

$$
I(X_i)=−\log_{2}{p(x_i)} \tag{1} 
$$

由于信息熵是信息量的期望值，所以信息熵`H(X)`的定义如下(其中`n`为数据集中类别的数量)：

$$
H(X)=−\sum_{i=1}^{n}p(x_i) \log_{2}{p(x_i)} \tag{2}
$$

从这个公式也可以看出，如果概率是`0`或者是`1`的时候，熵就是`0`（因为这种情况下随机变量的不确定性是最低的）。那如果概率是`0.5`，也就是五五开的时候，此时熵达到最大，也就是`1`。（就像扔硬币，你永远都猜不透你下次扔到的是正面还是反面，所以它的不确定性非常高）。所以呢，熵越大，不确定性就越高。

##### 条件熵

在实际的场景中，我们可能需要研究数据集中某个特征等于某个值时的信息熵等于多少，这个时候就需要用到条件熵。条件熵`H(Y|X)`表示特征 X 为某个值的条件下，类别为 Y 的熵。条件熵的计算公式如下：

$$
H(Y|X)= \sum_{i=1}^{n}p_iH(Y|X=x_i) \tag{3}
$$

当然条件熵的性质也和熵的性质一样，概率越确定，条件熵就越小，概率越五五开，条件熵就越大。

##### 信息增益

现在已经知道了什么是熵，什么是条件熵。接下来就可以看看什么是信息增益了。所谓的信息增益就是表示我已知条件`X`后能得到信息`Y`的不确定性的减少程度。

就好比，我在玩读心术。你心里想一件东西，我来猜。我已开始什么都没问你，我要猜的话，肯定是瞎猜。这个时候我的熵就非常高。然后我接下来我会去试着问你是非题，当我问了是非题之后，我就能减小猜测你心中想到的东西的范围，这样其实就是减小了我的熵。那么我熵的减小程度就是我的信息增益。

所以信息增益如果套上机器学习的话就是，如果把特征`A`对训练集`D`的信息增益记为`g(D, A)`的话，那么`g(D, A)`的计算公式就是：

$$
 g(D,A)=H(D)-H(D,A) \tag{4} 
$$

为了更好的解释熵，条件熵，信息增益的计算过程，下面通过示例来描述。假设我现在有这一个数据集，第一列是编号，第二列是性别，第三列是活跃度，第四列是客户是否流失的标签（`0`表示未流失，`1`表示流失）。

| 编号 | 性别 | 活跃度 | 是否流失 |
| ---- | ---- | ------ | -------- |
| 1    | 男   | 高     | 0        |
| 2    | 女   | 中     | 0        |
| 3    | 男   | 低     | 1        |
| 4    | 女   | 高     | 0        |
| 5    | 男   | 高     | 0        |
| 6    | 男   | 中     | 0        |
| 7    | 男   | 中     | 1        |
| 8    | 女   | 中     | 0        |
| 9    | 女   | 低     | 1        |
| 10   | 女   | 中     | 0        |
| 11   | 女   | 高     | 0        |
| 12   | 男   | 低     | 1        |
| 13   | 女   | 低     | 1        |
| 14   | 男   | 高     | 0        |
| 15   | 男   | 高     | 0        |

假如要算性别和活跃度这两个特征的信息增益的话，首先要先算总的熵和条件熵。总的熵其实非常好算，就是把标签作为随机变量`X`。上表中标签只有两种（`0`和`1`）因此随机变量`X`的取值只有`0`或者`1`。所以要计算熵就需要先分别计算标签为`0`的概率和标签为`1`的概率。从表中能看出标签为`0`的数据有`10`条，所以标签为`0`的概率等于`2/3`。标签为`1`的概率为`1/3`。所以熵为：

$$
−(1/3)log(1/3)−(2/3)log(2/3)=0.9182
$$

接下来就是条件熵的计算，以性别为男的熵为例。表格中性别为男的数据有`8`条，这`8`条数据中有`3`条数据的标签为`1`，有`5`条数据的标签为`0`。所以根据条件熵的计算公式能够得出该条件熵为：

$$
−(3/8)log(3/8)−(5/8)log(5/8)=0.9543
$$

根据上述的计算方法可知，总熵为：

$$
−(5/15)log(5/15)−(10/15)log(10/15)=0.9182
$$

性别为男的熵为：

$$
−(3/8)log(3/8)−(5/8)log(5/8)=0.9543
$$

性别为女的熵为：

$$
−(2/7)log(2/7)−(5/7)log(5/7)=0.8631
$$

活跃度为低的熵为：

$$
−(4/4)log(4/4)−0=0
$$

活跃度为中的熵为：

$$
−(1/5)log(1/5)−(4/5)log(4/5)=0.7219
$$

活跃度为高的熵为：

$$
−0−(6/6)log(6/6)=0
$$

现在有了总的熵和条件熵之后就能算出性别和活跃度这两个特征的信息增益了。

性别的信息增益=总的熵-(8/15)\*性别为男的熵-(7/15)\*性别为女的熵=0.0064

活跃度的信息增益=总的熵-(6/15)*活跃度为高的熵-(5/15)*活跃度为中的熵-(4/15)\*活跃度为低的熵=0.6776

那信息增益算出来之后有什么意义呢？回到读心术的问题，为了我能更加准确的猜出你心中所想，我肯定是问的问题越好就能猜得越准！换句话来说我肯定是要想出一个信息增益最大（减少不确定性程度最高）的问题来问你。其实`ID3`算法也是这么想的。`ID3`算法的思想是从训练集`D`中计算每个特征的信息增益，然后看哪个最大就选哪个作为当前结点。然后继续重复刚刚的步骤来构建决策树。

#### 编程要求

根据提示，在右侧编辑器补充代码，完成`calcInfoGain`函数实现计算信息增益。

`calcInfoGain`函数中的参数:

- `feature`：测试用例中字典里的`feature`，类型为`ndarray`；
- `label`：测试用例中字典里的`label`，类型为`ndarray`；
- `index`：测试用例中字典里的`index`，即`feature`部分特征列的索引。该索引指的是`feature`中第几个特征，如`index:0`表示使用第一个特征来计算信息增益。

#### 测试说明

平台会对你编写的代码进行测试，期望您的代码根据输入来输出正确的信息增益，以下为其中一个测试用例：

测试输入： `{'feature':[[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]], 'label':[0, 1, 0, 0, 1], 'index': 0}`

预期输出： `0.419973`

提示： 计算`log`可以使用`NumPy`中的`log2`函数

#### 参考答案

```python
import numpy as np


def calcInfoGain(feature, label, index):
    """
        计算信息增益
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:信息增益，类型float
    """

    # 计算熵
    def calcInfoEntropy(feature, label):
        """
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :return:信息熵，类型float
        """
        label_set = set(label)  # 创建一个无序不重复的元素集
        result = 0
        # 统计不同标签各自的数量（一般为0和1）
        for l in label_set:
            count = 0
            for j in range(len(label)):
                if label[j] == l:
                    count += 1
            # 计算标签在数据集中出现的概率
            p = count / len(label)
            # 计算熵
            result -= p * np.log2(p)
        return result

    # 计算条件熵
    def calcHDA(feature, label, index, value):
        """
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :param index:需要使用的特征列索引，类型为int
            :param value:index所表示的特征列中需要考察的特征值，类型为int
            :return:信息熵，类型float
        """
        count = 0
        # sub_feature和sub_label表示根据特征列和特征值
        # 分割出的子数据集中的特征和标签
        sub_feature = []
        sub_label = []
        for i in range(len(feature)):
            if feature[i][index] == value:
                count += 1
                sub_feature.append(feature[i])
                sub_label.append(label[i])
        pHA = count / len(feature)
        e = calcInfoEntropy(sub_feature, sub_label)
        return pHA * e

    #######请计算信息增益############
    # *********** Begin ***********#
    values = []  # 定义一个列表存放index列，即特征列的所有特征
    for i in range(len(feature)):
        values.append(feature[i][index])
    values_list = set(values)  # 创建一个无序不重复的元素集
    g = calcInfoEntropy(feature, label)  # 计算总熵
    for i in values_list:
        g -= calcHDA(feature, label, index, i)  # 总熵-每个特征的条件熵
    return g  # 得到信息增益
    # *********** End *************#

```

## 第 3 关：使用 ID3 算法构建决策树

#### 任务描述

本关任务：补充`python`代码，完成`DecisionTree`类中的`fit`和`predict`函数。

#### 相关知识

为了完成本关任务，你需要掌握：

- `ID3`算法构造决策树的流程；
- 如何使用构造好的决策树进行预测。

##### ID3 算法

`ID3`算法其实就是依据特征的信息增益来构建树的。其大致步骤就是从根结点开始，对结点计算所有可能的特征的信息增益，然后选择信息增益**最大**的特征作为结点的特征，由该特征的不同取值建立子结点，然后对子结点递归执行上述的步骤直到信息增益很小或者没有特征可以继续选择为止。

因此，`ID3`算法伪代码如下：

```python
# 假设数据集为D，标签集为A，需要构造的决策树为tree
def ID3(D, A):
    if D中所有的标签都相同:
        return 标签
    if 样本中只有一个特征或者所有样本的特征都一样:
        对D中所有的标签进行计数
        return 计数最高的标签

    计算所有特征的信息增益
    选出增益最大的特征作为最佳特征(best_feature)
    将best_feature作为tree的根结点
    得到best_feature在数据集中所有出现过的值的集合(value_set)
    for value in value_set:
        从D中筛选出best_feature = value的子数据集(sub_feature)
        从A中筛选出best_feature = value的子标签集(sub_label)
        # 递归构造tree
        tree[best_feature][value] = ID3(sub_feature, sub_label)
    return tree

```

##### 使用决策树进行预测

决策树的预测思想非常简单，假设现在已经构建出了一棵用来决策是否买西瓜的决策树。

![img-2](https://data.educoder.net/api/attachments/283157)

并假设现在在水果店里有这样一个西瓜，其属性如下：

| 瓤是否够红 | 够不够冰 | 是否便宜 | 是否有籽 |
| ---------- | -------- | -------- | -------- |
| 是         | 否       | 是       | 否       |

那买不买这个西瓜呢？只需把西瓜的属性代入决策树即可。决策树的根结点是`瓤是否够红`，所以就看西瓜的属性，经查看发现够红，因此接下来就看`够不够冰`。而西瓜不够冰，那么看`是否便宜`。发现西瓜是便宜的，所以这个西瓜是可以买的。

因此使用决策树进行预测的伪代码也比较简单，伪代码如下：

```python
#tree表示决策树，feature表示测试数据
def predict(tree, feature):
    if tree是叶子结点:
        return tree
    根据feature中的特征值走入tree中对应的分支
    if 分支依然是课树:
        result = predict(分支, feature)
    return result
```

#### 编程要求

填写`fit(self, feature, label)`函数，实现`ID3`算法，要求决策树保存在`self.tree`中。其中：

- `feature`：训练集数据，类型为`ndarray`，数值全为整数；
- `label`：训练集标签，类型为`ndarray`，数值全为整数。

填写`predict(self, feature)`函数，实现预测功能，并将标签返回，其中：

- `feature`：测试集数据，类型为`ndarray`，数值全为整数。**（PS：feature 中有多条数据）**

#### 测试说明

只需完成`fit`与`predict`函数即可，程序内部会调用您所完成的`fit`函数构建模型并调用`predict`函数来对数据进行预测。预测的准确率高于`0.92`视为过关。(PS:若`self.tree is None`则会打印**决策树构建失败**)

#### 参考答案

```python
import numpy as np


class DecisionTree(object):
    def __init__(self):
        # 决策树模型
        self.tree = {}

    def calcInfoGain(self, feature, label, index):
        """
            计算信息增益
            :param feature:测试用例中字典里的feature，类型为ndarray
            :param label:测试用例中字典里的label，类型为ndarray
            :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
            :return:信息增益，类型float
        """

        # 计算熵
        def calcInfoEntropy(label):
            """
                计算信息熵
                :param label:数据集中的标签，类型为ndarray
                :return:信息熵，类型float
            """
            label_set = set(label)
            result = 0
            for l in label_set:
                count = 0
                for j in range(len(label)):
                    if label[j] == l:
                        count += 1
                # 计算标签在数据集中出现的概率
                p = count / len(label)
                # 计算熵
                result -= p * np.log2(p)
            return result

        # 计算条件熵
        def calcHDA(feature, label, index, value):
            """
                计算信息熵
                :param feature:数据集中的特征，类型为ndarray
                :param label:数据集中的标签，类型为ndarray
                :param index:需要使用的特征列索引，类型为int
                :param value:index所表示的特征列中需要考察的特征值，类型为int
                :return:信息熵，类型float
            """
            count = 0
            # sub_feature和sub_label表示根据特征列和特征值分割出的子数据集中的特征和标签
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][index] == value:
                    count += 1
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            pHA = count / len(feature)
            e = calcInfoEntropy(sub_label)
            return pHA * e

        base_e = calcInfoEntropy(label)  # 信息熵
        f = np.array(feature)
        # 得到指定特征列的值的集合
        f_set = set(f[:, index])
        sum_HDA = 0
        # 计算条件熵
        for value in f_set:
            sum_HDA += calcHDA(feature, label, index, value)
        # 计算信息增益
        return base_e - sum_HDA

    # 获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        max_infogain = 0
        best_feature = 0
        # 每一列
        for i in range(len(feature[0])):
            infogain = self.calcInfoGain(feature, label, i)  # 计算每一个特征的信息增益
            if infogain > max_infogain:
                max_infogain = infogain
                best_feature = i
        return best_feature

    def createTree(self, feature, label):
        # 1.所有的标签相同，样本里都是同一个label没必要继续分叉了
        if len(set(label)) == 1:
            return label[0]
        # 2.样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的票数高
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:
            vote = {}
            # 为不同的label投票，计算数量最高的label
            for l in label:
                if l in vote.keys():
                    vote[l] += 1
                else:
                    vote[l] = 1
            # 求vote中计数最高的label
            max_count = 0
            vote_label = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    vote_label = k
            return vote_label
        # 3.第三种情况，根据信息增益拿到特征的索引
        best_feature = self.getBestFeature(feature, label)
        # 创建树，根结点为信息增益最大的特征索引
        tree = {best_feature: {}}
        f = np.array(feature)
        # 拿到bestfeature的所有特征值
        f_set = set(f[:, best_feature])
        # 构建对应特征值的子样本集sub_feature, sub_label
        for v in f_set:
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                # 在此特征的此样本条下构建子树
                if feature[i][best_feature] == v:
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            # 递归构建决策树
            tree[best_feature][v] = self.createTree(sub_feature, sub_label)
        return tree

    def fit(self, feature, label):
        """
            :param feature: 训练集数据，类型为ndarray
            :param label:训练集标签，类型为ndarray
            :return: None
        """
        # ************* Begin ************#
        self.tree = self.createTree(feature, label)
        # ************* End **************#

    def predict(self, feature):
        """
            :param feature:测试集数据，类型为ndarray
            :return:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        """

        # ************* Begin ************#

        def classify(tree, feature):
            # 如果tree是叶子结点，也就不是字典类型，返回结点
            if not isinstance(tree, dict):
                return tree
            # tree.items()返回可遍历的(键, 值) 元组数组。
            t_index, t_value = list(tree.items())[0]
            # t_index:树根特征对应feature的index,eg:feature[4,3,1,0]
            f_value = feature[t_index]  # 最优信息增益index对应的特征值
            if isinstance(t_value, dict):  # 如果tree是叶子结点，继续分叉
                value = classify(tree[t_index][f_value], feature)  # 递归此结点后面的树
                return value
            # 最后一个，叶子结点，对应的为标签值
            else:
                return t_value

        label = []
        for f in feature:
            # 添加通过决策树模型找到的叶子结点
            label.append(classify(self.tree, f))
        return np.array(label)
        # ************* End **************#

```

## 第 4 关：信息增益率

#### 任务描述

本关任务：根据本关所学知识，完成`calcInfoGainRatio`函数。

#### 相关知识

为了完成本关任务，你需要掌握：信息增益率

##### 信息增益率

由于在使用信息增益这一指标进行划分时，更喜欢可取值数量较多的特征。为了减少这种**偏好**可能带来的不利影响，`Ross Quinlan`使用了**信息增益率**这一指标来选择最优划分属性。

信息增益率的数学定义为如下，其中*D*表示数据集，*a*表示数据集中的某一列，_G**a**i\*\*n_(_D_,_a_)表示*D*中*a*的信息增益，*V*表示*a*这一列中取值的集合，*v*表示*V*中的某种取值，∣*D*∣ 表示*D*中样本的数量，∣*D\*\*v*∣ 表示*D*中*a*这一列中值等于*v*的数量。

$$
Gain\_ratio(D,a)=\frac{Gain(D,a)}{-\sum_{v=1}^{V}\log_{2}{\frac{|D^v|}{|D|}}} \tag{5}
$$

从公式可以看出，信息增益率很好算，只是用信息增益除以另一个分母，该分母通常称为**固有值**。举个例子，还是使用**第二关**中提到过的数据集，第一列是编号，第二列是性别，第三列是活跃度，第四列是客户是否流失的标签（`0`表示未流失，`1`表示流失）。

| 编号 | 性别 | 活跃度 | 是否流失 |
| ---- | ---- | ------ | -------- |
| 1    | 男   | 高     | 0        |
| 2    | 女   | 中     | 0        |
| 3    | 男   | 低     | 1        |
| 4    | 女   | 高     | 0        |
| 5    | 男   | 高     | 0        |
| 6    | 男   | 中     | 0        |
| 7    | 男   | 中     | 1        |
| 8    | 女   | 中     | 0        |
| 9    | 女   | 低     | 1        |
| 10   | 女   | 中     | 0        |
| 11   | 女   | 高     | 0        |
| 12   | 男   | 低     | 1        |
| 13   | 女   | 低     | 1        |
| 14   | 男   | 高     | 0        |
| 15   | 男   | 高     | 0        |

根据**第二关**已经知道性别的信息增益为 0.0064，设*a*为性别，则有**Gain**(_D_,_a_)=0.0064。由根据数据可知，_V_=2，假设当*v*=1 时表示性别为男，_v_=2 时表示性别为女，则有 ∣*D*∣=15，∣*D*1∣=8，∣*D*2∣=7。因此根据信息增益率的计算公式可知**$Gain_{ratio}$**(_D_,_a_)=0.0642。同理可以算出活跃度的信息增益率为 0.4328。

#### 编程要求

根据提示，在右侧编辑器补充代码，完成`calcInfoGainRatio`函数实现计算信息增益。

`calcInfoGainRatio`函数中的参数:

- `feature`：测试用例中字典里的`feature`，类型为`ndarray`；
- `label`：测试用例中字典里的`label`，类型为`ndarray`；
- `index`：测试用例中字典里的`index`，即`feature`部分特征列的索引。该索引指的是`feature`中第几个特征，如`index:0`表示使用第一个特征来计算信息增益率。

#### 测试说明

平台会对你编写的代码进行测试，期望您的代码根据输入来输出正确的信息增益，以下为其中一个测试用例：

测试输入： `{'feature':[[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]], 'label':[0, 1, 0, 0, 1], 'index': 0}`

预期输出： `0.432538`

提示： 计算`log`可以使用`NumPy`中的`log2`函数

#### 参考答案

```python
import numpy as np


def calcInfoGain(feature, label, index):
    """
        计算信息增益
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:信息增益，类型float
    """

    # 计算熵
    def calcInfoEntropy(label):
        """
            计算信息熵
            :param label:数据集中的标签，类型为ndarray
            :return:信息熵，类型float
        """

        label_set = set(label)
        result = 0
        for l in label_set:
            count = 0
            for j in range(len(label)):
                if label[j] == l:
                    count += 1
            # 计算标签在数据集中出现的概率
            p = count / len(label)
            # 计算熵
            result -= p * np.log2(p)
        return result

    # 计算条件熵
    def calcHDA(feature, label, index, value):
        """
            计算信息熵
            :param feature:数据集中的特征，类型为ndarray
            :param label:数据集中的标签，类型为ndarray
            :param index:需要使用的特征列索引，类型为int
            :param value:index所表示的特征列中需要考察的特征值，类型为int
            :return:信息熵，类型float
        """
        count = 0
        # sub_label表示根据特征列和特征值分割出的子数据集中的标签
        sub_label = []
        for i in range(len(feature)):
            if feature[i][index] == value:
                count += 1
                sub_label.append(label[i])
        pHA = count / len(feature)
        e = calcInfoEntropy(sub_label)
        return pHA * e

    base_e = calcInfoEntropy(label)
    f = np.array(feature)
    # 得到指定特征列的值的集合,:表示获取所有行
    f_set = set(f[:, index])  # 将不重复的特征值获取出来（比如:男，女）
    sum_HDA = 0
    # 计算条件熵
    for value in f_set:
        sum_HDA += calcHDA(feature, label, index, value)
    # 计算信息增益
    return base_e - sum_HDA


def calcInfoGainRatio(feature, label, index):
    """
        计算信息增益率
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:信息增益率，类型float
    """

    # ********* Begin *********#
    up = calcInfoGain(feature, label, index)  # 信息增益率的分子

    # 定义一个方法求分母中某个类型的个数(如求当v=1时表示性别为男的)
    def dcon(feature, value):
        s = 0
        for i in range(len(feature)):
            if feature[i][index] == value:
                s += 1
            else:
                pass
        return s

    down = 0
    # 取出特征值该列所有数据
    values = []
    for i in range(len(feature)):
        values.append(feature[i][index])
    values_set = set(values)  # 使用set()过滤重复值，得到特征值列中所有类型(如性别中男和女)
    # 循环递归求出分母
    for value in values_set:
        down -= (dcon(feature, value) / len(feature)) * np.log2(dcon(feature, value) / len(feature))
    # 求得信息增益率
    gain = up / down
    return gain
    # ********* End *********#

```

## 第 5 关：基尼系数

#### 任务描述

本关任务：根据本关所学知识，完成`calcGini`函数。

#### 相关知识

为了完成本关任务，你需要掌握：基尼系数。

##### 基尼系数

在`ID3`算法中我们使用了信息增益来选择特征，信息增益大的优先选择。在`C4.5`算法中，采用了信息增益率来选择特征，以减少信息增益容易选择特征值多的特征的问题。但是无论是`ID3`还是`C4.5`,都是基于信息论的熵模型的，这里面会涉及大量的对数运算。能不能简化模型同时也不至于完全丢失熵模型的优点呢？当然有！那就是**基尼系数**！

`CART`算法使用**基尼系数**来代替信息增益率，基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。这和信息增益与信息增益率是相反的(它们都是越大越好)。

基尼系数的数学定义为如下，其中*D*表示数据集，*p\*\*k*表示`D`中第`k`个类别在`D`中所占比例。

Gini(D)=1−sum\k=1∣y∣pk*2

从公式可以看出，相比于信息增益和信息增益率，计算起来更加简单。举个例子，还是使用**第二关**中提到过的数据集，第一列是编号，第二列是性别，第三列是活跃度，第四列是客户是否流失的标签（`0`表示未流失，`1`表示流失）。

| 编号 | 性别 | 活跃度 | 是否流失 |
| ---- | ---- | ------ | -------- |
| 1    | 男   | 高     | 0        |
| 2    | 女   | 中     | 0        |
| 3    | 男   | 低     | 1        |
| 4    | 女   | 高     | 0        |
| 5    | 男   | 高     | 0        |
| 6    | 男   | 中     | 0        |
| 7    | 男   | 中     | 1        |
| 8    | 女   | 中     | 0        |
| 9    | 女   | 低     | 1        |
| 10   | 女   | 中     | 0        |
| 11   | 女   | 高     | 0        |
| 12   | 男   | 低     | 1        |
| 13   | 女   | 低     | 1        |
| 14   | 男   | 高     | 0        |
| 15   | 男   | 高     | 0        |

从表格可以看出，*D*中总共有 2 个类别，设类别为 0 的比例为*p*1，则有*p*1=1510。设类别为 1 的比例为*p*2，则有*p*2=155。根据基尼系数的公式可知*G**i**n\*\*i*(_D_)=1−(*p*12+*p*22)=0.4444。

上面是基于数据集`D`的基尼系数的计算方法，那么基于数据集`D`与特征`a`的基尼系数怎样计算呢？其实和信息增益率的套路差不多。计算公式如下：

$$ Gini(D,a)=\sum\_{v=1}^{V}\frac{|D^v|}{|D|}Gini(D^v) \tag6 $$

还是以用户流失的数据为例，现在算一算性别的基尼系数。设性别男为*v*=1，性别女为*v*=2 则有 ∣*D*∣=15，∣*D*1∣=8，∣*D*2∣=7，_G**i**n\*\*i_(*D*1)=0.46875，_G**i**n\*\*i_(*D*2)=0.40816。所以*G**i**n\*\*i*(_D_,_a_)=0.44048。

#### 编程要求

根据提示，在右侧编辑器补充代码，完成`calcGini`函数实现计算信息增益。

`calcGini`函数中的参数:

- `feature`：测试用例中字典里的`feature`，类型为`ndarray`；
- `label`：测试用例中字典里的`label`，类型为`ndarray`；
- `index`：测试用例中字典里的`index`，即`feature`部分特征列的索引。该索引指的是`feature`中第几个特征，如`index:0`表示使用第一个特征来计算基尼系数。

#### 测试说明

平台会对你编写的代码进行测试，期望您的代码根据输入来输出正确的信息增益，以下为其中一个测试用例：

测试输入： `{'feature':[[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]], 'label':[0, 1, 0, 0, 1], 'index': 0}`

预期输出： `0.266667`

#### 参考答案

```python
import numpy as np


def calcGini(feature, label, index):
    """
        计算基尼系数
        :param feature:测试用例中字典里的feature，类型为ndarray
        :param label:测试用例中字典里的label，类型为ndarray
        :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
        :return:基尼系数，类型float
    """

    # ********* Begin *********#
    def _gini(label):
        unique_label = list(set(label))
        gini = 1
        for l in unique_label:
            p = np.sum(label == l) / len(label)
            gini -= p ** 2
        return gini

    unique_value = list(set(feature[:, index]))
    gini = 0
    for value in unique_value:
        len_v = np.sum(feature[:, index] == value)
        gini += (len_v / len(feature)) * _gini(label[feature[:, index] == value])
    return gini
    # ********* End *********#

```

## 第 6 关：预剪枝与后剪枝

#### 任务描述

本关任务：补充`python`代码，完成`DecisionTree`类中的`fit`和`predict`函数。

#### 相关知识

为了完成本关任务，你需要掌握：

- 为什么需要剪枝；
- 预剪枝；
- 后剪枝。

##### 为什么需要剪枝

决策树的生成是递归地去构建决策树，直到不能继续下去为止。这样产生的树往往对训练数据有很高的分类准确率，但对未知的测试数据进行预测就没有那么准确了，也就是所谓的过拟合。

决策树容易过拟合的原因是在构建决策树的过程时会过多地考虑如何提高对训练集中的数据的分类准确率，从而会构建出非常复杂的决策树（树的宽度和深度都比较大）。在之前的实训中已经提到过，**模型的复杂度越高，模型就越容易出现过拟合的现象。**所以简化决策树的复杂度能够有效地缓解过拟合现象，而简化决策树最常用的方法就是剪枝。剪枝分为预剪枝与后剪枝。

##### 预剪枝

预剪枝的核心思想是在决策树生成过程中，对每个结点在划分前先进行一个评估，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点。

想要评估决策树算法的泛化性能如何，方法很简单。可以将训练数据集中随机取出一部分作为验证数据集，然后在用训练数据集对每个结点进行划分之前用当前状态的决策树计算出在验证数据集上的正确率。正确率越高说明决策树的泛化性能越好，如果在划分结点的时候发现泛化性能有所下降或者没有提升时，说明应该停止划分，并用投票计数的方式将当前结点标记成叶子结点。

举个例子，假如上一关中所提到的用来决定是否买西瓜的决策树模型已经出现过拟合的情况，模型如下：

![img-3](https://data.educoder.net/api/attachments/283157)

假设当模型在划分`是否便宜`这个结点前，模型在验证数据集上的正确率为`0.81`。但在划分后，模型在验证数据集上的正确率降为`0.67`。此时就不应该划分`是否便宜`这个结点。所以预剪枝后的模型如下：

![img-4](https://data.educoder.net/api/attachments/283551)

从上图可以看出，**预剪枝能够降低决策树的复杂度。这种预剪枝处理属于贪心思想，但是贪心有一定的缺陷，就是可能当前划分会降低泛化性能，但在其基础上进行的后续划分却有可能导致性能显著提高。所以有可能会导致决策树出现欠拟合的情况。**

##### 后剪枝

后剪枝是先从训练集生成一棵完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能够带来决策树泛化性能提升，则将该子树替换为叶结点。

后剪枝的思路很直接，对于决策树中的每一个非叶子结点的子树，我们尝试着把它替换成一个叶子结点，该叶子结点的类别我们用子树所覆盖训练样本中存在最多的那个类来代替，这样就产生了一个简化决策树，然后比较这两个决策树在测试数据集中的表现，如果简化决策树在验证数据集中的准确率有所提高，那么该子树就可以替换成叶子结点。该算法以`bottom-up`的方式遍历所有的子树，直至没有任何子树可以替换使得测试数据集的表现得以改进时，算法就可以终止。

从后剪枝的流程可以看出，后剪枝是从全局的角度来看待要不要剪枝，所以造成欠拟合现象的可能性比较小。但由于后剪枝需要先生成完整的决策树，然后再剪枝，所以后剪枝的训练时间开销更高。

#### 编程要求

填写`fit(self, train_feature, train_label, val_featrue, val_label)`函数，实现带**后剪枝**的`ID3`算法，要求决策树保存在`self.tree`中。其中：

- `train_feature`：训练集数据，类型为`ndarray`，数值全为整数；
- `train_label`：训练集标签，类型为`ndarray`，数值全为整数；
- `val_feature`：验证集数据，类型为`ndarray`，数值全为整数；
- `val_label`：验证集标签，类型为`ndarray`，数值全为整数。

填写`predict(self, feature)`函数，实现预测功能，并将标签返回，其中：

- `feature`：测试集数据，类型为`ndarray`，数值全为整数。**（PS：feature 中有多条数据）**

#### 测试说明

只需完成`fit`与`predict`函数即可，程序内部会调用您所完成的`fit`函数构建模型并调用`predict`函数来对数据进行预测。预测的准确率高于`0.935`视为过关。(PS:若`self.tree is None`则会打印**决策树构建失败**)

#### 参考答案

```python
import numpy as np
from copy import deepcopy


class DecisionTree(object):
    def __init__(self):
        # 决策树模型
        self.tree = {}

    def calcInfoGain(self, feature, label, index):
        """
            计算信息增益
            :param feature:测试用例中字典里的feature，类型为ndarray
            :param label:测试用例中字典里的label，类型为ndarray
            :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
            :return:信息增益，类型float
        """

        # 计算熵
        def calcInfoEntropy(feature, label):
            """
                计算信息熵
                :param feature:数据集中的特征，类型为ndarray
                :param label:数据集中的标签，类型为ndarray
                :return:信息熵，类型float
            """

            label_set = set(label)
            result = 0
            for l in label_set:
                count = 0
                for j in range(len(label)):
                    if label[j] == l:
                        count += 1
                # 计算标签在数据集中出现的概率
                p = count / len(label)
                # 计算熵
                result -= p * np.log2(p)
            return result

        # 计算条件熵
        def calcHDA(feature, label, index, value):
            """
                计算信息熵
                :param feature:数据集中的特征，类型为ndarray
                :param label:数据集中的标签，类型为ndarray
                :param index:需要使用的特征列索引，类型为int
                :param value:index所表示的特征列中需要考察的特征值，类型为int
                :return:信息熵，类型float
            """
            count = 0
            # sub_feature和sub_label表示根据特征列和特征值分割出的子数据集中的特征和标签
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][index] == value:
                    count += 1
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            pHA = count / len(feature)
            e = calcInfoEntropy(sub_feature, sub_label)
            return pHA * e

        base_e = calcInfoEntropy(feature, label)
        f = np.array(feature)
        # 得到指定特征列的值的集合
        f_set = set(f[:, index])
        sum_HDA = 0
        # 计算条件熵
        for value in f_set:
            sum_HDA += calcHDA(feature, label, index, value)
        # 计算信息增益
        return base_e - sum_HDA

    # 获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        max_infogain = 0
        best_feature = 0
        for i in range(len(feature[0])):
            infogain = self.calcInfoGain(feature, label, i)
            if infogain > max_infogain:
                max_infogain = infogain
                best_feature = i
        return best_feature

    # 计算验证集准确率
    def calc_acc_val(self, the_tree, val_feature, val_label):
        result = []

        def classify(tree, feature):
            if not isinstance(tree, dict):
                return tree
            t_index, t_value = list(tree.items())[0]
            f_value = feature[t_index]
            if isinstance(t_value, dict):
                classLabel = classify(tree[t_index][f_value], feature)
                return classLabel
            else:
                return t_value

        for f in val_feature:
            result.append(classify(the_tree, f))

        result = np.array(result)
        return np.mean(result == val_label)

    def createTree(self, train_feature, train_label):
        # 样本里都是同一个label没必要继续分叉了
        if len(set(train_label)) == 1:
            return train_label[0]
        # 样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的票数高
        if len(train_feature[0]) == 1 or len(np.unique(train_feature, axis=0)) == 1:
            vote = {}
            for l in train_label:
                if l in vote.keys():
                    vote[l] += 1
                else:
                    vote[l] = 1
            max_count = 0
            vote_label = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    vote_label = k
            return vote_label

        # 根据信息增益拿到特征的索引
        best_feature = self.getBestFeature(train_feature, train_label)
        tree = {best_feature: {}}
        f = np.array(train_feature)
        # 拿到bestfeature的所有特征值
        f_set = set(f[:, best_feature])
        # 构建对应特征值的子样本集sub_feature, sub_label
        for v in f_set:
            sub_feature = []
            sub_label = []
            for i in range(len(train_feature)):
                if train_feature[i][best_feature] == v:
                    sub_feature.append(train_feature[i])
                    sub_label.append(train_label[i])

            # 递归构建决策树
            tree[best_feature][v] = self.createTree(sub_feature, sub_label)

        return tree

    # 后剪枝
    def post_cut(self, val_feature, val_label):
        # 拿到非叶子节点的数量
        def get_non_leaf_node_count(tree):
            non_leaf_node_path = []

            def dfs(tree, path, all_path):
                for k in tree.keys():
                    if isinstance(tree[k], dict):
                        path.append(k)
                        dfs(tree[k], path, all_path)
                        if len(path) > 0:
                            path.pop()
                    else:
                        all_path.append(path[:])

            dfs(tree, [], non_leaf_node_path)

            unique_non_leaf_node = []
            for path in non_leaf_node_path:
                isFind = False
                for p in unique_non_leaf_node:
                    if path == p:
                        isFind = True
                        break
                if not isFind:
                    unique_non_leaf_node.append(path)
            return len(unique_non_leaf_node)

        # 拿到树中深度最深的从根节点到非叶子节点的路径
        def get_the_most_deep_path(tree):
            non_leaf_node_path = []

            def dfs(tree, path, all_path):
                for k in tree.keys():
                    if isinstance(tree[k], dict):
                        path.append(k)
                        dfs(tree[k], path, all_path)
                        if len(path) > 0:
                            path.pop()
                    else:
                        all_path.append(path[:])

            dfs(tree, [], non_leaf_node_path)

            max_depth = 0
            result = None
            for path in non_leaf_node_path:
                if len(path) > max_depth:
                    max_depth = len(path)
                    result = path
            return result

        # 剪枝
        def set_vote_label(tree, path, label):
            for i in range(len(path) - 1):
                tree = tree[path[i]]
            tree[path[len(path) - 1]] = vote_label

        acc_before_cut = self.calc_acc_val(self.tree, val_feature, val_label)
        # 遍历所有非叶子节点
        for _ in range(get_non_leaf_node_count(self.tree)):
            path = get_the_most_deep_path(self.tree)

            # 备份树
            tree = deepcopy(self.tree)
            step = deepcopy(tree)

            # 跟着路径走
            for k in path:
                step = step[k]

            # 叶子节点中票数最多的标签
            vote_label = sorted(step.items(), key=lambda item: item[1], reverse=True)[0][0]

            # 在备份的树上剪枝
            set_vote_label(tree, path, vote_label)

            acc_after_cut = self.calc_acc_val(tree, val_feature, val_label)

            # 验证集准确率高于0.9才剪枝
            if acc_after_cut > acc_before_cut:
                set_vote_label(self.tree, path, vote_label)
                acc_before_cut = acc_after_cut

    def fit(self, train_feature, train_label, val_feature, val_label):
        """
            :param train_feature:训练集数据，类型为ndarray
            :param train_label:训练集标签，类型为ndarray
            :param val_feature:验证集数据，类型为ndarray
            :param val_label:验证集标签，类型为ndarray
            :return: None
        """

        # ************* Begin ************#
        self.tree = self.createTree(train_feature, train_label)
        # 后剪枝
        self.post_cut(val_feature, val_label)
        # ************* End **************#

    def predict(self, feature):
        """
            :param feature:测试集数据，类型为ndarray
            :return:预测结果，如np.array([0, 1, 2, 2, 1, 0])
        """

        # ************* Begin ************#
        result = []

        # 单个样本分类
        def classify(tree, feature):
            if not isinstance(tree, dict):
                return tree
            t_index, t_value = list(tree.items())[0]
            f_value = feature[t_index]
            if isinstance(t_value, dict):
                classLabel = classify(tree[t_index][f_value], feature)
                return classLabel
            else:
                return t_value

        for f in feature:
            result.append(classify(self.tree, f))

        return np.array(result)
        # ************* End **************#

```

## 第 7 关：鸢尾花识别

#### 任务描述

本关任务：使用`sklearn`完成鸢尾花分类任务。

#### 相关知识

为了完成本关任务，你需要掌握如何使用`sklearn`提供的`DecisionTreeClassifier`。

##### 数据简介

![img-6](https://data.educoder.net/api/attachments/283552)

鸢尾花数据集是一类多重变量分析的数据集。通过花萼长度，花萼宽度，花瓣长度，花瓣宽度`4`个属性预测鸢尾花卉属于(`Setosa`，`Versicolour`，`Virginica`)三个种类中的哪一类(其中分别用`0`，`1`，`2`代替)。

数据集中部分数据与标签如下图所示：

![img-7](https://data.educoder.net/api/attachments/317817)

![img-7](https://data.educoder.net/api/attachments/317819)

##### DecisionTreeClassifier

`DecisionTreeClassifier`的构造函数中有两个常用的参数可以设置：

- `criterion`:划分节点时用到的指标。有`gini`（**基尼系数**）,`entropy`(**信息增益**)。若不设置，默认为`gini`
- `max_depth`:决策树的最大深度，如果发现模型已经出现过拟合，可以尝试将该参数调小。若不设置，默认为`None`

和`sklearn`中其他分类器一样，`DecisionTreeClassifier`类中的`fit`函数用于训练模型，`fit`函数有两个向量输入：

- `X`：大小为`[样本数量,特征数量]`的`ndarray`，存放训练样本；
- `Y`：值为整型，大小为`[样本数量]`的`ndarray`，存放训练样本的分类标签。

`DecisionTreeClassifier`类中的`predict`函数用于预测，返回预测标签，`predict`函数有一个向量输入：

- `X`：大小为`[样本数量,特征数量]`的`ndarray`，存放预测样本。

`DecisionTreeClassifier`的使用代码如下：

```python
from sklearn.tree import DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
result = clf.predict(X_test)
```

数据文件格式如下图所示:

![img-8](https://data.educoder.net/api/attachments/317828)

标签文件格式如下图所示:

![img-9](https://data.educoder.net/api/attachments/317829)

**PS：`predict.csv`文件的格式必须与标签文件格式一致。**

#### 测试说明

只需将结果保存至`./step7/predict.csv`即可，程序内部会检测您的代码，预测准确率高于`0.95`视为过关。

#### 参考答案

```python
# ********* Begin *********#
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('./step7/train_data.csv').as_matrix()
train_label = pd.read_csv('./step7/train_label.csv').as_matrix()
test_df = pd.read_csv('./step7/test_data.csv').as_matrix()

dt = DecisionTreeClassifier()
dt.fit(train_df, train_label)
result = dt.predict(test_df)

result = pd.DataFrame({'target': result})
result.to_csv('./step7/predict.csv', index=False)
# ********* End *********#

```

