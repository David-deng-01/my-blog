---
title: 深度学习代码笔记-01
layout: post
tags:
  - 笔记
  - 深度学习
categories:
  - 笔记
  - DL
lang: zh-CN
abbrlink: 43177
date: 2023-05-05 21:47:43
---

# 深度学习代码笔记-01

## 1. 配置环境

## 1.1 `Conda`

> 任选其一（推荐后者）
>
> 1. [Anaconda 安装](https://repo.anaconda.com/archive/)
> 2. [Miniconda 安装](https://docs.conda.io/en/latest/miniconda.html)

## 1.2 Conda 常用命令

```shell
# 显示所有环境
conda env list

# 显示当前环境下的包
conda list

# 创建conda环境
conda create -n 环境名 python=版本号

# 删除conda环境
conda remove -n 环境名 --all

# 进入conda环境
conda activate 环境名

# 退出conda环境
conda deactivate

# 删除缓存
conda clean -a -y

# conda环境导出
conda activate 环境名
conda env export > env.yaml

# conda环境迁移
conda env create -f env.yaml

# conda国内源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

## 1.2 安装`Pytorch`

官网地址：[PyTorch](https://pytorch.org/)

![pythorch.png](https://cdn.jsdelivr.net/gh/David-deng-01/images/blog/image-20230506004259724.png)

## 1. 分词任务

### 任务简介：

> 模型内部是一系列的矩阵运算，只能处理数字。因此倘若需要让模型处理一个句子（比如判断这个句子是积极的，还是消极的），需要先把句子转为一串数字。所以在 NLP 学习中，我们需要先了解怎么将文本进行分词，并将每一个词都转化成对应的词向量。

### 任务步骤

1. 安装第三方库`pip install numpy nltk transformers`
2. 下载词向量文件 `glove.6B.50d.txt`, 下载地址：[glove.6B.50d.txt](https://www.kaggle.com/datasets/watts2/glove6b50dtxt)
3. 任务目标：将每个词转为词向量

### Example 1 代码解释

1. 首先需要导入需要的第三方依赖

   ```python
   # 如果没有安装第三方依赖, 请安装
   # pip install numpy nltk transformers
   from typing import Dict, List

   import numpy as np
   from nltk import word_tokenize
   from numpy import ndarray
   ```

2. 加载词向量文件

   - 从`glove.6B.50d.txt` 文件中按行读取词向量，每次读取一行
   - 按照空格分割每一行的数据
   - 分割得到的列表(list) 第一个元素是单词, 后面所有的元素是单词对应的词向量(vector)
   - 将单词(word)作为 key, 词向量(vector) 作为 value, 存入 result 中

   ```python
   def read_glove_file(self) -> Dict[str, List[float]]:
       """
           读取glove词向量文件, 并将其转换为字典形式
           :return:  Dict[str, List[float]], key 为词, value 为词向量
       """
       result: Dict[str, List[float]] = {}
       glove_path = f"{self.base_path}/{self.glove_file_name}"
       print(f'加载词向量文件：{glove_path}')
       with open(glove_path, 'r', encoding='utf-8') as file:
           while True:
               line: str = file.readline()  # 读取一行
               if not line:
                   break  # 如果没有读取成功
               else:
                   line_split: list[str] = line.strip().split()  # 按空格分割读取的一行数据
                   word: str = line_split[0]  # 第一个为词，作为 key
                   vector: list[float] = list(map(float, line_split[1:]))  # 除了第一个元素外, 其他元素组成对应的词向量
                   result[word] = vector  # 将词作为 key, 向量作为 value, 存入结果中
       return result
   ```

3. 将句子转换成对应的词向量

   ```python
   def run(self) -> ndarray:
       print(f'原始句子：{self.sentence}')

       # 第一步：分词
       tokens: list[str] = word_tokenize(self.sentence)
       print(f'分词结果：{tokens}')
       print(f'序列长度：{len(tokens)}')

       # 第二步：加载glove词向量文件, 提取每个word的词向量
       word_to_vector_dict: Dict[str, List[float]] = self.read_glove_file()
       dimension: int = len(word_to_vector_dict['the'])
       print(f'词向量的大小：{len(word_to_vector_dict)}')
       print(f"单词的维度：{dimension}")

       special_token_list: list[str] = ['unk', 'pad', 'cls', 'sep']
       for sp_token in special_token_list:
           if sp_token not in word_to_vector_dict:  # 如果特殊字符不在词向量文件中
               word_to_vector_dict[sp_token] = np.random.random(dimension).tolist()  # 随机生成一些数字放入词向量文件中,作为特殊字符的词向量

       # 第三步：转为词向量
       arr = []
       for token in tokens:
           # 将分词得到的 token 通过词向量表, 转换成对应的词向量
           if token not in word_to_vector_dict:
               arr.append(word_to_vector_dict['unk'])
           else:
               arr.append(word_to_vector_dict[token])

       vector: ndarray = np.array(arr)  # 将数组转成 numpy.ndarray
       print(f'数组形状为：{vector.shape}')

       # 返回分词的结果
       return vector
   ```

   **`操作步骤：`**

   1. 将句子(sentence)分词, 分成一个一个的单词(word)
   2. 将单词(word)通过 `word_to_vector_dict` 转换成对应的词向量 (vector)
   3. 将所有单词(word)的词向量(vector)按照顺序放入 `arr` 列表中，然后将 `arr` 数据类型转换成 `numpy.ndarray`

4. 完整代码: `Example1.py`

   ```python
   from typing import Dict, List
   
   import numpy as np
   from nltk import word_tokenize
   from numpy import ndarray
   
   
   class Example1(object):
       def __init__(self):
           # 原始的句子
           self.sentence = 'Commonsense knowledge and commonsense reasoning play ' \
                           'a vital role in all aspects of machine intelligence,' \
                           'from language understanding to computer vision and ' \
                           'robotics .'.lower()
           self.base_path = "Model/glove"  # 基础路径
           self.glove_file_name = "glove.6B.50d.txt"  # 词向量文件名
   
       def run(self) -> ndarray:
           print(f'原始句子：{self.sentence}')
   
           # 第一步：分词
           tokens: list[str] = word_tokenize(self.sentence)
           print(f'分词结果：{tokens}')
           print(f'序列长度：{len(tokens)}')
   
           # 第二步：加载glove词向量文件, 提取每个word的词向量
           word_to_vector_dict: Dict[str, List[float]] = self.read_glove_file()
           dimension: int = len(word_to_vector_dict['the'])
           print(f'词向量的大小：{len(word_to_vector_dict)}')
           print(f"单词的维度：{dimension}")
   
           special_token_list: list[str] = ['unk', 'pad', 'cls', 'sep']
           for sp_token in special_token_list:
               if sp_token not in word_to_vector_dict:  # 如果特殊字符不在词向量文件中
                   word_to_vector_dict[sp_token] = np.random.random(dimension).tolist()  # 随机生成一些数字放入词向量文件中,作为特殊字符的词向量
   
           # 第三步：转为词向量
           arr = []
           for token in tokens:
               # 将分词得到的 token 通过词向量表, 转换成对应的词向量
               if token not in word_to_vector_dict:
                   arr.append(word_to_vector_dict['unk'])
               else:
                   arr.append(word_to_vector_dict[token])
   
           vector: ndarray = np.array(arr)  # 将数组转成 numpy.ndarray
           print(f'数组形状为：{vector.shape}')
   
           # 返回分词的结果
           return vector
   
       def read_glove_file(self) -> Dict[str, List[float]]:
           """
               读取glove词向量文件, 并将其转换为字典形式
               :return:  Dict[str, List[float]], key 为词, value 为词向量
           """
           result: Dict[str, List[float]] = {}
           glove_path = f"{self.base_path}/{self.glove_file_name}"
           print(f'加载词向量文件：{glove_path}')
           with open(glove_path, 'r', encoding='utf-8') as file:
               while True:
                   line: str = file.readline()  # 读取一行
                   if not line:
                       break  # 如果没有读取成功
                   else:
                       line_split: list[str] = line.strip().split()  # 按空格分割读取的一行数据
                       word: str = line_split[0]  # 第一个为词，作为 key
                       vector: list[float] = list(map(float, line_split[1:]))  # 除了第一个元素外, 其他元素组成对应的词向量
                       result[word] = vector  # 将词作为 key, 向量作为 value, 存入结果中
           return result
   
   
   if __name__ == '__main__':
       v = Example1().run()
       print(v)
   
   ```

   


   `代码运行结果：`

   <img src="img/index/image-20230507233228291.png" alt="image-20230507233228291" style="zoom:100%;" />

### Example2 代码解释

1. `Example2.py` 和 `Example1.py `的区别：

   1. `Example2.py` 是 `Example1.py` 的升级版，`Example1.py` 只是演示了如果将句子分词，将单词通过词向量文件转化成对应的词向量
   2. `Example2.py` 是我们在写模型中真正会用到的分词过程，不仅仅是将句子转换成对应的词向量
   3. 这两个文件最大的区别就是，在 `Example2.py` 中我们构建了自己的词汇表，然后将输入的句子通过词汇表转换成了对应的 token id

2. 首先需要先了解一下词汇表这个类 `Vocabulary`

   1. `Vocabulary` 类中有 5 个主要的成员变量，分别是: `id_to_word: dict[int, str]`、`word_to_id: dict[str, int]`、`word_feq: defaultdict[str, int]`、`special_token_list: list[str]`和`size: int`

   2. 初始化方法 `def __init__(self)` 如下：

      ```python
      def __init__(self) -> None:
          self.word_to_id: Dict[str, int] = {}  # key: 单词, value: 单词 id
          self.id_to_word: Dict[int, str] = {}  # key: 单词 id, value: 单词
          self.word_feq: defaultdict[str, int] = defaultdict(int)  # 单词的频繁程度, 单词在词汇表中出现的次数
          self.special_token_list: List[str] = []  # 特殊的 token
          self.size: int = 0  # 词汇表的大小, 初始时为 0
          self.save_path = "vocabulary"  # 默认的文件保存路径
          self.load_path = self.save_path  # 默认的文件加载路径
          self.keys = ["word_to_id", "id_to_word", "special_token_list", "word_feq",
                       "size"]  # 保存时 json 文件的key, 也是 Vocabulary 的属性名称
      ```
   
   3. `Vocabulary` 类中剩下的部分就是对这些变量的操作，详细的请看代码中的注释，具体代码如下：
   
      ```python
      # 词汇表类
      import json
      import os
      from collections import defaultdict
      from typing import Dict, List
      
      
      class Vocabulary(object):
          # 成员变量的类型提示
          id_to_word: dict[int, str]
          word_to_id: dict[str, int]
          word_feq: defaultdict[str, int]
          special_token_list: list[str]
          size: int
      
          def __init__(self) -> None:
              self.word_to_id: Dict[str, int] = {}  # key: 单词, value: 单词 id
              self.id_to_word: Dict[int, str] = {}  # key: 单词 id, value: 单词
              self.word_feq: defaultdict[str, int] = defaultdict(int)  # 单词的频繁程度, 单词在词汇表中出现的次数
              self.special_token_list: List[str] = []  # 特殊的 token
              self.size: int = 0  # 词汇表的大小, 初始时为 0
              self.save_path = "vocabulary"  # 默认的文件保存路径
              self.load_path = self.save_path  # 默认的文件加载路径
              self.keys = ["word_to_id", "id_to_word", "special_token_list", "word_feq",
                           "size"]  # 保存时 json 文件的key, 也是 Vocabulary 的属性名称
      
          def add_token(self, token: str) -> int:
              """
                  将当前 token 添加到当前词汇表中
                  :param token: 待添加的 token
                  :return: 添加 token 后对应的 token id
              """
              if token not in self.word_to_id:
                  self.word_to_id[token] = self.size  # 将 token 添加到 word_to_id 中
                  self.id_to_word[self.size] = token  # 将 token 添加到 id_to_word 中
                  self.size += 1  # 词汇表中 token 数量 +1
              self.word_feq[token] += 1  # 当前 token 出现的频率 +1
              return self.word_to_id[token]  # 返回当前 token 的 id
      
          def add_tokens(self, tokens: List[str]) -> List[int]:
              """
                  批量添加 token 到词汇表中
                  :param tokens: 待添加的 token 列表
                  :return: 添加 token 后对应的 token id
              """
              return list(self.add_token(token) for token in tokens)
      
          def add_special_token(self, token: str) -> int:
              """
                  添加特殊的 token 或者说添加自定义的 token
                  :param token: 待添加的 token
                  :return: 添加 token 后对应的 token id
              """
              if token not in self.special_token_list:
                  # 如果 token 没有在特殊 token 列表中出现
                  self.special_token_list.append(token)  # 将当前 token 添加到特殊 token 列表中
                  return self.add_token(token)  # 将当前 token 添加到词汇表中, 并返回添加后的 token id
              else:
                  # 如果 token 已经在特殊 token 列表中出现过
                  return self.word_to_id[token]  # 查询词汇表, 返回对应的 token id
      
          def add_special_tokens(self, tokens: List[str]) -> List[int]:
              """
                  批量添加特殊的 token 到词汇表中
                  :param tokens: 待添加的 token 列表
                  :return: 添加 token 后对应的 token id
              """
              return list(self.add_special_token(token) for token in tokens)
      
          def save_vocabulary_to_file(self, path: str = "", filename: str = "vocabulary.json") -> None:
              """
                  保存 Vocabulary 到指定的路径下
                  :param filename: 默认文件名称s
                  :param path: 指定的保存路径
                  :return: None
              """
              if path == "":
                  path = self.save_path
      
              os.makedirs(path, exist_ok=True)
      
              vocabulary_dict: dict = {key: getattr(self, key) for key in self.keys}  # 通过属性名称获取对应的属性值
              with open(f"{path}/{filename}", "w", encoding="utf-8") as file:
                  json.dump(vocabulary_dict, file)  # 将 Vocabulary 类中指定的属性写入文件中
      
          def load_vocabulary_from_file(self, path: str = "", filename: str = "vocabulary.json") -> None:
              """
                  读取指定路径下的 Vocabulary
                  :param filename: 默认文件名称
                  :param path:  指定的读取路径
                  :return: None
              """
              if path == "":
                  path = self.load_path
      
              if not os.path.exists(path):
                  # 文件路径不存在, 直接返回
                  print(f"【{path}/{filename}】, 文件不存在")
                  return
      
              with open(f"{path}/{filename}", "r", encoding="utf-8") as file:
                  vocabulary_dict: dict = json.load(file)  # 读取文件中的词汇表
      
              for key in self.keys:
                  setattr(self, key, vocabulary_dict[key])  # 通过 key 为 Vocabulary 的属性进行赋值
      
          def covert_token_to_id(self, token: str) -> int:
              """
                  将 token 转换成 token id
                  :param token: 待转换的 token
                  :return: 指定 token 的 token id
              """
              return self.word_to_id[token]
      
          def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
              """
                  批量 将 token 转换成 token id
                  :param tokens: 待转换的 token 列表
                  :return: token 列表中 token 的 token id
              """
              return list(self.covert_token_to_id(token) for token in tokens)
      
          def convert_id_to_token(self, token_id: int) -> str:
              """
                  通过 id 获取指定的 token
                  :param token_id: token id
                  :return: 指定的 token
              """
              return self.id_to_word[token_id]
      
          def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
              """
                  批量通过 id 获取指定的 token
                  :param token_ids: token id 列表
                  :return: 指定的 token 列表
              """
              return list(self.convert_id_to_token(token_id) for token_id in token_ids)
      ```


3. 首先我们需要根据自己的数据集创建对应的词汇表，这里演示使用的是 `Yelp` 数据集，详细的数据集可以在 [Hugging Face](https://huggingface.co/) 中下载，也可以使用我已经下载好的 [Yelp 数据集](https://cdn.jsdelivr.net/gh/David-deng-01/images/dataset/yelp.jsonl) 数据集里面的数据大致格式如下图所示：

   <img src="https://cdn.jsdelivr.net/gh/David-deng-01/images/blog/image-20230508172028197.png" alt="image-20230508172028197" style="zoom:100%;" />

   `text` 表示句子， `label` 是句子的标签, 0 表示句子情感消极, 1 表示句子情感积极

4. 根据数据集(dataset)创建词汇表(vocabulary)具体操作步骤如下:

   1. 按行读取数据集，将句子取出
   2. 将句子进行分词，每个单词都存入词汇表中

   创建词汇表的代码如下：

   ```python
   def create_my_vocabulary(data_file_path: str = "") -> Vocabulary:
       """
           根据自己的数据集, 创建自己的词汇表
           :param data_file_path: 词汇表路径
           :return:
       """
       voca = Vocabulary()  # 创建词汇表对象
       if data_file_path == "":
           # 如果没有传入数据集保存的位置, 则加载默认的词汇表
           voca.load_vocabulary_from_file()
           return voca

       voca.add_special_tokens(['unk', 'pad', 'cls', 'sep'])  # 向词汇表中添加特殊 token
       with open(data_file_path, "r", encoding="utf-8") as file:
           for line in file.readlines():
               text: str = json.loads(line)['text'].lower()  # 提取文件中的所有句子, 并将句子中的单词全部小写
               voca.add_tokens(nltk.word_tokenize(text))  # 使用分词工具, 将句子分词后添加到词汇表中
       return voca
   ```

5. 词汇表创建结束后，再将词向量文件加载到模型中，具体的加载操作与 `Example1.py` 中加载词向量文件的操作相似。具体代码如下：

   ```python
   def read_glove(self) -> Dict[str, FloatTensor]:
       """
           读取预训练的词向量文件
           :return: 单词和对应的词向量
       """
       print("开始读取词向量文件".center(50, "*"))
       word_to_vector: Dict[str, FloatTensor] = {}
       with open(self.glove_path, "r", encoding="utf-8") as file:
           while True:
               line = file.readline()  # 从词向量文件中读取一行数据
               if not line:
                   break  # 如果没有读取到数据, 即 line is None, 跳出循环
               else:
                   line_split: list[str] = line.strip().split()  # 删除 line 前面和后面多余的空格, 并将 line 按照空格分开
                   word: str = line_split[0]  # 单词, str
                   vector: FloatTensor = FloatTensor(list(map(float, line_split[1:])))  # 单词对应的向量, FloatTensor
                   word_to_vector[word] = vector  # key: word, value: vector
       print("词向量文件读取完成".center(50, "*"))
       return word_to_vector
   ```

6. 接下来是创建模型，我们使用的是 `torch.nn.Embedding` 如果没有安装 `Pytorch` 请先安装，再进行接下来的操作

7. 创建模型后，将一个批量(batch) 的数据放入模型中，在实际的神经网络训练中，我们也是一个批量一个批量的将数据送入模型中，而不是一条一条数据送入模型中

8. 将数据输入模型后，在模型内部会进行填充(padding)操作，这是因为模型一般都是对矩阵进行操作，但是我们输入的句子可能有长有短，所以需要将短的句子使用特殊的 token 填充到和当前 batch 中最长的句子一样长，这样模型才能进行处理

9. `Example2.py` 完整代码如下：

   ```python
   import json
   import os
   from typing import List, Dict
   
   import nltk
   import torch
   from torch import nn as nn, FloatTensor, LongTensor
   from torch.nn import Embedding
   
   from utils.Vocabulary import Vocabulary
   
   
   def create_my_vocabulary(data_file_path: str = "") -> Vocabulary:
       """
           根据自己的数据集, 创建自己的词汇表
           :param data_file_path: 词汇表路径
           :return:
       """
       voca = Vocabulary()  # 创建词汇表对象
       if data_file_path == "":
           # 如果没有传入数据集保存的位置, 则加载默认的词汇表
           voca.load_vocabulary_from_file()
           return voca
   
       voca.add_special_tokens(['unk', 'pad', 'cls', 'sep'])  # 向词汇表中添加特殊 token
       with open(data_file_path, "r", encoding="utf-8") as file:
           for line in file.readlines():
               text: str = json.loads(line)['text'].lower()  # 提取文件中的所有句子, 并将句子中的单词全部小写
               voca.add_tokens(nltk.word_tokenize(text))  # 使用分词工具, 将句子分词后添加到词汇表中
       return voca
   
   
   class Example2(nn.Module):
       def __init__(self, vocabulary: Vocabulary = None, dimension: int = 50):
           """
               初始化方法
               :param vocabulary: 词汇表
               :param dimension: 每个单词的维度, 默认为 50 维
           """
           # 调用父类 nn.Module 的初始化方法
           super(Example2, self).__init__()
   
           # 保存词汇表
           self.vocabulary: Vocabulary = vocabulary
   
           # 创建一个嵌入层 embedding, 是一个二维矩阵, 形状为 (word_number, dimension)
           self.embedding: Embedding = nn.Embedding(num_embeddings=vocabulary.size, embedding_dim=dimension)
   
           # 路径
           self.glove_path = "Model/glove/glove.6B.50d.txt"  # 预训练的词向量文件位置
           self.output_dir = "output/example2"  # 输出文件保存位置
           self.vocabulary_filename = "vocabulary.json"  # 输出的词汇表文件名称
           self.vocabulary_ckpt_path = f"{self.output_dir}/{self.vocabulary_filename}"  # 词汇表保存位置
   
       def init_embedding(self) -> None:
           """
               初始化
               :return: None
           """
           # 1.加载预训练的词向量文件
           word_to_vector = self.read_glove()
   
           # 2. 计算词向量文件相对于自己的数据集的命中率
           hit = 0  # 命中次数
           unhit_token = []  # 未命中的 token
           for word, word_id in self.vocabulary.word_to_id.items():
               if word in word_to_vector:
                   hit += 1  # 命中的次数 +1
                   vector = word_to_vector[word]  # 获取当前 word 对应的 vector
                   self.embedding.weight.data[word_id] = vector  # 给词向量赋值
               else:
                   unhit_token.append(word)
           print(f"由数据库创建的单词表的大小为: {self.vocabulary.size}")
           print(f"其中{hit}个词有预训练的词向量, 命中率为: {hit / self.vocabulary.size}")
           print(f"没有预训练词向量的词有: {unhit_token}, 它们的词向量是随机初始化的")
   
       def read_glove(self) -> Dict[str, FloatTensor]:
           """
               读取预训练的词向量文件
               :return: 单词和对应的词向量
           """
           print("开始读取词向量文件".center(50, "*"))
           word_to_vector: Dict[str, FloatTensor] = {}
           with open(self.glove_path, "r", encoding="utf-8") as file:
               while True:
                   line = file.readline()  # 从词向量文件中读取一行数据
                   if not line:
                       break  # 如果没有读取到数据, 即 line is None, 跳出循环
                   else:
                       line_split: list[str] = line.strip().split()  # 删除 line 前面和后面多余的空格, 并将 line 按照空格分开
                       word: str = line_split[0]  # 单词, str
                       vector: FloatTensor = FloatTensor(list(map(float, line_split[1:])))  # 单词对应的向量, FloatTensor
                       word_to_vector[word] = vector  # key: word, value: vector
           print("词向量文件读取完成".center(50, "*"))
           return word_to_vector
   
       def forward(self, input_text: List[str]) -> FloatTensor:
           """
               查询 input_text 中的句子的
               :param input_text: 一个个句子组成的列表
               :return:
           """
           # 1. 将句子全部进行分词转为相应 token id
           word_list: list[list[str]] = list(nltk.word_tokenize(sentence) for sentence in input_text)
           input_ids: list[list[int]] = list(self.vocabulary.convert_tokens_to_ids(sequence) for sequence in word_list)
   
           # 2. 计算序列的最大长度, 方便以后的 padding 操作, 因为模型输入的是矩阵, 如果句子的长度不一样, 我们应该进行 padding 操作
           max_len: int = max(len(sequence) for sequence in input_ids)
   
           # 3. 获取用于 padding 的 token id, 即 "pad" 的 id
           pad_id: int = self.vocabulary.word_to_id["pad"]
   
           # 4. 因为 input_ids 中 list 的长度不一, 所以需要统一长度, 即 padding 操作
           input_ids: list[list[int]] = list(sequence + [pad_id] * (max_len - len(sequence)) for sequence in input_ids)
   
           # 5. 将 input_ids 转成 tensor, shape >> [bath_size, max_len]
           input_ids: LongTensor = LongTensor(input_ids)
   
           # 6. 将 input_ids 转成词向量
           input_embedding: FloatTensor = self.embedding(input_ids)
   
           return input_embedding
   
   
   def run(dimension: int = 50):
       data_file_path = "data/yelp.jsonl"  # 自己的数据集的位置
       output_dir = "output/example2"  # 输出文件保存位置
       embedding_layer_ckpt_path = f"{output_dir}/embedding_layer.pt"  # 嵌入层保存位置
   
       # 1. 创建输出文件目录, 如果不存在的话
       os.makedirs(output_dir, exist_ok=True)
   
       # 2. 根据自己的数据集创建词汇表
       voca = create_my_vocabulary(data_file_path)
   
       # 3. 创建 embedding 层
       embedding_layer = Example2(vocabulary=voca, dimension=dimension)
       if embedding_layer_ckpt_path is None:
           embedding_layer.init_embedding()
           torch.save(embedding_layer.state_dict(), f"{output_dir}/embedding_layer.pt")
       else:
           embedding_layer.load_state_dict(torch.load(embedding_layer_ckpt_path))
   
       # 4. 将一个 batch 的 sentences 转为向量
       # 演示的batch size = 4
       batch_text = [
           "ever since joes has changed hands it 's just gotten worse and worse .",
           "there is definitely not enough room in that part of the venue .",
           "so basically tasted watered down .",
           "she said she 'd be back and disappeared for a few minutes ."
       ]
       print([item.shape for item in embedding_layer.forward(batch_text)])
       print(embedding_layer(batch_text).shape)
   
   
   if __name__ == '__main__':
       run(dimension=50)
   
   ```


   `代码运行结果如下`：

<img src="https://cdn.jsdelivr.net/gh/David-deng-01/images/blog/image-20230508180244621.png" alt="image-20230508180244621" style="zoom:100%;" />

## 2. 句子情感分类任务

## 3. 对话生成任务


