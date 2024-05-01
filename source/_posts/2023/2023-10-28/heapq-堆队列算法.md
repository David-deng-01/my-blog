---
title: heapq 堆队列算法
layout: post
tags:
  - python
  - heapq
categories:
  - python
  - heapq
lang: zh-CN
abbrlink: 50914
date: 2023-10-28 13:34:19
---

# Python `heapq `堆队列算法

## 模块解释说明

源码：[Lib/heapq.py](https://github.com/python/cpython/tree/3.12/Lib/heapq.py)

> 这个模块实现了堆队列算法，即优先队列算法。
>
> 堆是一棵完全二叉树，其中每个节点的值都小于等于其各个子节点的值。这个使用数组的实现，索引从 0 开始，且对所有的 *k* 都有 `heap[k] <= heap[2*k+1]` 和 `heap[k] <= heap[2*k+2]`。比较时不存在的元素被认为是无限大。堆最有趣的特性在于最小的元素总是在根结点：`heap[0]`。
>
> 这个API与教材的堆算法实现有所不同，具体区别有两方面：（a）我们使用了从零开始的索引。这使得节点和其孩子节点索引之间的关系不太直观但更加适合，因为 Python 使用从零开始的索引。 （b）我们的 pop 方法返回最小的项而不是最大的项（这在教材中称为“最小堆”；而“最大堆”在教材中更为常见，因为它更适用于原地排序）。
>
> 基于这两方面，把堆看作原生的Python list也没什么奇怪的： `heap[0]` 表示最小的元素，同时 `heap.sort()` 维护了堆的不变性！
>
> 要创建一个堆，可以新建一个空列表 `[]`，或者用函数 [`heapify()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heapify) 把一个非空列表变为堆。

总结如下：

1. `heapq` 模块实现了堆队列算法(优先队列算法)，进一步说，`heapq` 模块实现了小顶堆(小根堆)
2. 介绍了堆的基本概念，堆是一个完全二叉树，满足对于所有的 *k* 都有 `heap[k] <= heap[2*k+1]` 和 `heap[k] <= heap[2*k+2]`。
3. 由于`heapq` 模块实现的是小根堆，并且`heap[0]`表示根节点，所以可以得出结论: `heap[0]` 是最小的元素。
4. 堆可以看作原生的Python list：`heap[0]` 表示最小的元素，同时 `heap.sort()` 维护了堆的不变性！要创建一个堆，可以新建一个空列表 `[]`，或者用函数 [`heapify()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heapify) 把一个非空列表变为堆。



## `heapq` 模块中的函数

### `heapq.heapify(x)`

> 将list *x* 转换成堆，原地，线性时间内。

示例代码：

```python
from heapq import heapify
hq = [1, 4, 56, 2, 3, 66, 12, 98, 34, 6, 35]  # 创建一个 Python list

print(f"转换前, list hq 为 >> {hq}")
heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)
print(f"转换后, 堆队列为 >> {hq}")

"""
运行结果:
转换前, list hq 为 >> [1, 4, 56, 2, 3, 66, 12, 98, 34, 6, 35]
转换后, 堆队列为 >> [1, 2, 12, 4, 3, 66, 56, 98, 34, 6, 35]
"""
```



### `heapq.heappush(heap, item)`

>  将 *item* 的值加入 *heap* 中，保持堆的不变性。

示例代码：

```python
from heapq import heappush, heapify


hq = [1, 4, 56, 2, 3, 66, 12, 98, 34, 6, 35]  # 创建一个 Python list

heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)

print(f"添加元素前, 堆队列为 >> {hq}")
heappush(hq, 88)  # 将 item 的值加入 hq 中，保持堆的不变性
print(f"添加元素后, 堆队列为 >> {hq}")

"""
运行结果:
添加元素前, 堆队列为 >> [1, 2, 12, 4, 3, 66, 56, 98, 34, 6, 35]
添加元素后, 堆队列为 >> [1, 2, 12, 4, 3, 66, 56, 98, 34, 6, 35, 88]
"""
```



### `heapq.heappop(heap)`

> 弹出并返回 *heap* 的最小的元素，保持堆的不变性。如果堆为空，抛出 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError) 。使用 `heap[0]` ，可以只访问最小的元素而不弹出它。

示例代码：

```python
from heapq import heappop, heapify


hq = [1, 4, 56, 2, 3, 66, 12, 98, 34, 6, 35]  # 创建一个 Python list

heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)
print(f"堆队列为 >> {hq}")
print(f"弹出前, 根节点元素为 >> {hq[0]}")

res = heappop(hq) # 弹出并返回 hq 的最小的元素，保持堆的不变性
print(f"弹出元素为 >> {hq[0]}")
print(f"弹出后, 根节点元素为 >> {hq[0]}")

"""
运行结果:
堆队列为 >> [1, 2, 12, 4, 3, 66, 56, 98, 34, 6, 35]
弹出前, 根节点元素为 >> 1
弹出元素为 >> 2
弹出后, 根节点元素为 >> 2
"""
```



### `heapq.heappushpop(heap, item)`

> 将 *item* 放入堆中，然后弹出并返回 *heap* 的最小元素。该组合操作比先调用 [`heappush()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappush) 再调用 [`heappop()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappop) 运行起来更有效率。

示例代码：

```python
from heapq import heappushpop, heapify


hq = [1, 4, 56, 2, 3, 66, 12, 98, 34, 6, 35]  # 创建一个 Python list

heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)
print(f"插入前, 堆队列为 >> {hq}")
print(f"弹出前, 根节点元素为 >> {hq[0]}")

res = heappushpop(hq, 88) # 将 item 放入堆中，然后弹出并返回 hq 的最小元素
print(f"弹出元素为 >> {res}")
print(f"插入并弹出根节点后, 堆队列为 >> {hq}")
print(f"弹出后, 根节点元素为 >> {hq[0]}")

"""
运行结果:
插入前, 堆队列为 >> [1, 2, 12, 4, 3, 66, 56, 98, 34, 6, 35]
弹出前, 根节点元素为 >> 1
弹出元素为 >> 1
插入并弹出根节点后, 堆队列为 >> [2, 3, 12, 4, 6, 66, 56, 98, 34, 88, 35]
弹出后, 根节点元素为 >> 2
"""
```



### `heapq.heapreplace(heap, item)`

> 弹出并返回 *heap* 中最小的一项，同时推入新的 *item*。 堆的大小不变。 如果堆为空则引发 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError)。
>
> 这个单步骤操作比 [`heappop()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappop) 加 [`heappush()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappush) 更高效，并且在使用固定大小的堆时更为适宜。 pop/push 组合总是会从堆中返回一个元素并将其替换为 *item*。
>
> 返回的值可能会比新加入的值大。如果不希望如此，可改用 [`heappushpop()`](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heappushpop)。它的 push/pop 组合返回两个值中较小的一个，将较大的留在堆中。

示例代码：

```python
from heapq import heapreplace, heapify

hq = [1, 4, 56, 2, 3, 66, 12, 98, 34, 6, 35]  # 创建一个 Python list

heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)
print(f"插入前, 堆队列为 >> {hq}")
print(f"弹出前, 根节点元素为 >> {hq[0]}")

res = heapreplace(hq, 88) # 弹出并返回 hq 中最小的一项，同时推入新的 item
print(f"弹出元素为 >> {res}")
print(f"插入并弹出根节点后, 堆队列为 >> {hq}")
print(f"弹出后, 根节点元素为 >> {hq[0]}")

"""
运行结果:
插入前, 堆队列为 >> [1, 2, 12, 4, 3, 66, 56, 98, 34, 6, 35]
弹出前, 根节点元素为 >> 1
弹出元素为 >> 1
插入并弹出根节点后, 堆队列为 >> [2, 3, 12, 4, 6, 66, 56, 98, 34, 88, 35]
弹出后, 根节点元素为 >> 2
"""
```



## `heapq` 模块三个基于堆的通用目的函数 

### `heapq.merge(iterables, key=None, reverse=False)`

> 将多个`已排序的输入`合并为一个`已排序的输出`（例如，合并来自多个日志文件的带时间戳的条目）。 返回已排序值的 [iterator](https://docs.python.org/zh-cn/3/glossary.html#term-iterator)。
>
> 类似于 `sorted(itertools.chain(*iterables))` 但返回一个可迭代对象，不会一次性地将数据全部放入内存，并假定每个输入流都是已排序的（从小到大）。
>
> 具有两个可选参数，它们都必须指定为关键字参数。
>
> *key* 指定带有单个参数的 [key function](https://docs.python.org/zh-cn/3/glossary.html#term-key-function)，用于从每个输入元素中提取比较键。 默认值为 `None` (直接比较元素)。
>
> *reverse* 为一个布尔值。 如果设为 `True`，则输入元素将按比较结果逆序进行合并。 要达成与 `sorted(itertools.chain(*iterables), reverse=True)` 类似的行为，所有可迭代对象必须是已从大到小排序的。
>
> *在 3.5 版更改:* 添加了可选的 *key* 和 *reverse* 形参。

示例代码：

```python
import random
from heapq import  merge

"""
代码功能描述：
	1. 随机生成 4 个数组，每个数组包括 5 个数，数组中每个数的取值都在 [-100, 100] 之间
	2. 将生成的数组按照绝对值从小到大进行排序
	3. 使用 heapq.merge 函数合并 4 个有序的数组
注意:
	sorted 函数中的参数 key 和 reverse 必须和 heapq.merge 中的参数 key 和 reverse 相同
	如果不同，则会导致排序失败，排序后的数组将是无序的数组。
	从 heapq.merge 函数的描述中我们可以找到为啥 sorted 函数中的参数 key 和 reverse 必须和 heapq.merge 中的参数 key 和 reverse 相同
	因为 heapq.merge 函数是将多个 **已排序的输入** 合并为一个 **已排序的输出** ，返回已排序值的 iterator。
"""
min_value, max_value, row, column = -100, 100, 4, 5
key, reverse = lambda x: x if x >= 0 else -x, False
data = [
    sorted(list(random.sample(range(min_value, max_value + 1), column)), key=key, reverse=reverse)
    for _ in range(row)
]
for i, d in enumerate(data):
    print(f"{i}: {d}")
print(f"Merged: >> {list(merge(*data, key=key, reverse=reverse))}")

"""s
运行结果:
0: [8, -9, 37, 60, -76]
1: [16, -18, -28, 43, 64]
2: [-6, 19, -41, -52, 77]
3: [32, -75, -85, -91, 91]
Merged: >> [-6, 8, -9, 16, -18, 19, -28, 32, 37, -41, 43, -52, 60, 64, -75, -76, 77, -85, -91, 91]
"""
```



### `heapq.nlargest(n, iterable, key=None)`

> 从 *iterable* 所定义的数据集中返回前 *n* 个最大元素组成的列表。 如果提供了 *key* 则其应指定一个单参数的函数，用于从 *iterable* 的每个元素中提取比较键 (例如 `key=str.lower`)。 等价于: `sorted(iterable, key=key, reverse=True)[:n]`。

示例代码：

```python
import random
from heapq import nlargest, heapify


"""
代码功能描述：
	1. 随机生成 1 个包括 5 个数的数组，每个数的取值都在 [-100, 100] 之间
	2. 将数组转换成堆队列
	3. 取出堆队列中绝对值最大的 3 个值
"""
min_value, max_value, count,n = -100, 100, 10,3
hq = random.sample(range(min_value, max_value + 1), count)  # 创建一个 Python list
key = lambda a: a if a >= 0 else -a

heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)
largest = nlargest(n, hq, key=key)

print(f"堆队列为 >> {hq}")
print(f"绝对值最大的 {n} 个数 >> {largest}")

"""
运行结果:
堆队列为 >> [-86, -61, -6, -33, 8, 11, 97, 88, 82, 62]
绝对值最大的 3 个数 >> [97, 88, -86]
"""
```



### `heapq.nsmallest(n, iterable, key=None)`

> 从 *iterable* 所定义的数据集中返回前 *n* 个最小元素组成的列表。 如果提供了 *key* 则其应指定一个单参数的函数，用于从 *iterable* 的每个元素中提取比较键 (例如 `key=str.lower`)。 等价于: `sorted(iterable, key=key)[:n]`。

示例代码：

```python
import random
from heapq import nsmallest, heapify


"""
代码功能描述：
	1. 随机生成 1 个包括 5 个数的数组，每个数的取值都在 [-100, 100] 之间
	2. 将数组转换成堆队列
	3. 取出堆队列中绝对值最小的 3 个值
"""
min_value, max_value, count,n = -100, 100, 10,3
hq = random.sample(range(min_value, max_value + 1), count)  # 创建一个 Python list
key = lambda a: a if a >= 0 else -a

heapify(hq)  # 将 hq 转换成堆队列(构建小顶堆)
smallest = nsmallest(n, hq, key=key)

print(f"堆队列为 >> {hq}")
print(f"绝对值最小的 {n} 个数 >> {smallest}")

"""
运行结果:
堆队列为 >> [-86, -61, -6, -33, 8, 11, 97, 88, 82, 62]
绝对值最小的 3 个数 >> [-6, 8, 11]
"""
```

### 注意

1. `heapq.nlargest`和`heapq.nsmallest`函数在 *n* 值较小时性能最好。 
2. 对于更大的值，使用 [`sorted()`](https://docs.python.org/zh-cn/3/library/functions.html#sorted) 函数会更有效率。
3. 当 `n==1` 时，使用内置的 [`min()`](https://docs.python.org/zh-cn/3/library/functions.html#min) 和 [`max()`](https://docs.python.org/zh-cn/3/library/functions.html#max) 函数会更有效率。
4.  如果需要重复使用这些函数，请考虑将可迭代对象转为真正的堆。

## 基本示例

[堆排序](https://en.wikipedia.org/wiki/Heapsort) 可以通过将所有值推入堆中然后每次弹出一个最小值项来实现。

```python
import random
from heapq import heappush, heappop, heapify


def heap_sort(iterable):
    hq = []
    heapify(hq)
    for value in iterable:
        heappush(hq, value)
    return [heappop(hq) for i in range(len(hq))]


if __name__ == '__main__':
    min_value, max_value, count = -100, 100, 10
    arr = random.sample(range(min_value, max_value + 1), count)  # 创建一个 Python list
    print(f"堆排序前的结果 >> {arr}")
    print(f"堆排序后的结果 >> {heap_sort(arr)}")
 
"""
运行结果:
堆排序前的结果 >> [-87, 81, 100, 56, 10, 67, -19, -14, -7, -40]
堆排序后的结果 >> [-87, -40, -19, -14, -7, 10, 56, 67, 81, 100]
"""
```

这类似于 `sorted(iterable)`，但与 [`sorted()`](https://docs.python.org/zh-cn/3/library/functions.html#sorted) 不同的是这个实现是不稳定的。

堆元素可以为元组。这有利于以下做法——在被跟踪的主记录旁边添一个额外的值（例如任务的优先级）用于互相比较：

```python
from heapq import heappop, heapify


arr = [(5, 'write code'), (7, 'release product'), (1, 'write spec'), (3, 'create tests')] # 创建一个 list
heapify(arr) # 将 list 转换成小顶堆
print(heappop(arr)) # 输出小顶堆的根节点

"""
运行结果:
(1, 'write spec')
"""
```



## 优先队列实现说明

[优先队列](https://en.wikipedia.org/wiki/Priority_queue) 是堆的常用场合，并且它的实现包含了多个挑战：

- 排序稳定性：如何让两个相同优先级的任务按它们最初被加入队列的顺序返回？
- 如果 priority 相同且 task 之间未定义默认比较顺序，则两个 (priority, task) 元组之间的比较会报错。
- 如果任务优先级发生改变，你该如何将其移至堆中的新位置？
- 或者如果一个挂起的任务需要被删除，你该如何找到它并将其移出队列？

针对前两项挑战的一种解决方案是将条目保存为包含优先级、条目计数和任务对象 3 个元素的列表。 条目计数可用来打破平局，这样具有相同优先级的任务将按它们的添加顺序返回。 并且由于没有哪两个条目计数是相同的，元组比较将永远不会直接比较两个任务。

两个 task 之间不可比的问题的另一种解决方案是——创建一个忽略 task，只比较 priority 字段的包装器类：

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
```

其余的挑战主要包括找到挂起的任务并修改其优先级或将其完全移除。 找到一个任务可使用一个指向队列中条目的字典来实现。

移除条目或改变其优先级的操作实现起来更为困难，因为它会破坏堆结构不变量。 因此，一种可能的解决方案是将条目标记为已移除，再添加一个改变了优先级的新条目:

```python
pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task
counter = itertools.count()     # unique sequence count

def add_task(task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heappush(pq, entry)

def remove_task(task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while pq:
        priority, count, task = heappop(pq)
        if task is not REMOVED:
            del entry_finder[task]
            return task
    raise KeyError('pop from an empty priority queue')
```



## 理论

堆是通过数组来实现的，其中的元素从 0 开始计数，对于所有的 *k* 都有 `a[k] <= a[2*k+1]` 且 `a[k] <= a[2*k+2]`。 为了便于比较，不存在的元素被视为无穷大。 堆最有趣的特性在于 `a[0]` 总是其中最小的元素。

上面的特殊不变量是用来作为一场锦标赛的高效内存表示。 下面的数字是 *k* 而不是 `a[k]`:

```python
                               0

              1                                 2

      3               4                5               6

  7       8       9       10      11      12      13      14

15 16   17 18   19 20   21 22   23 24   25 26   27 28   29 30
```

在上面的树中，每个 *k* 单元都位于 `2*k+1` 和 `2*k+2` 之上。 体育运动中我们经常见到二元锦标赛模式，每个胜者单元都位于另两个单元之上，并且我们可以沿着树形图向下追溯胜者所遇到的所有对手。 但是，在许多采用这种锦标赛模式的计算机应用程序中，我们并不需要追溯胜者的历史。 为了获得更高的内存利用效率，当一个胜者晋级时，我们会用较低层级的另一条目来替代它，因此规则变为一个单元和它之下的两个单元包含三个不同条目，上方单元“胜过”了两个下方单元。

如果此堆的不变性质始终受到保护，则序号 0 显然是总的赢家。 删除它并找出“下一个”赢家的最简单算法方式是将某个输家（让我们假定是上图中的 30 号单元）移至 0 号位置，然后将这个新的 0 号沿树下行，不断进行值的交换，直到不变性质得到重建。 这显然会是树中条目总数的对数。 通过迭代所有条目，你将得到一个 O(n log n) 复杂度的排序。

此排序有一个很好的特性就是你可以在排序进行期间高效地插入新条目，前提是插入的条目不比你最近取出的 0 号元素“更好”。 这在模拟上下文时特别有用，在这种情况下树保存的是所有传入事件，“胜出”条件是最小调度时间。 当一个事件将其他事件排入执行计划时，它们的调试时间向未来方向延长，这样它们可方便地入堆。 因此，堆结构很适宜用来实现调度器，我的 MIDI 音序器就是用的这个 :-)。

用于实现调度器的各种结构都得到了充分的研究，堆是非常适宜的一种，因为它们的速度相当快，并且几乎是恒定的，最坏的情况与平均情况没有太大差别。 虽然还存在其他总体而言更高效的实现方式，但其最坏的情况却可能非常糟糕。

堆在大磁盘排序中也非常有用。 你应该已经了解大规模排序会有多个“运行轮次”（即预排序的序列，其大小通常与 CPU 内存容量相关），随后这些轮次会进入合并通道，轮次合并的组织往往非常巧妙 [1](https://docs.python.org/zh-cn/3/library/heapq.html#id2)。 非常重要的一点是初始排序应产生尽可能长的运行轮次。 锦标赛模式是达成此目标的好办法。 如果你使用全部有用内存来进行锦标赛，替换和安排恰好适合当前运行轮次的条目，你将可以对于随机输入生成两倍于内存大小的运行轮次，对于模糊排序的输入还会有更好的效果。

另外，如果你输出磁盘上的第 0 个条目并获得一个可能不适合当前锦标赛的输入（因为其值要“胜过”上一个输出值），它无法被放入堆中，因此堆的尺寸将缩小。 被释放的内存可以被巧妙地立即重用以逐步构建第二个堆，其增长速度与第一个堆的缩减速度正好相同。 当第一个堆完全消失时，你可以切换新堆并启动新的运行轮次。 这样做既聪明又高效！

总之，堆是值得了解的有用内存结构。 我在一些应用中用到了它们，并且认为保留一个 'heap' 模块是很有意义的。 :-)

## 异常解释

### `exception IndexError`

> 当序列抽取超出范围时将被引发。 （切片索引会被静默截短到允许的范围；如果指定索引不是整数则 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError) 会被引发。）



### `exception TypeError`

> 当一个操作或函数被应用于类型不适当的对象时将被引发。 关联的值是一个字符串，给出有关类型不匹配的详情。
>
> 此异常可以由用户代码引发，以表明尝试对某个对象进行的操作不受支持也不应当受支持。 如果某个对象应当支持给定的操作但尚未提供相应的实现，所要引发的适当异常应为 [`NotImplementedError`](https://docs.python.org/zh-cn/3/library/exceptions.html#NotImplementedError)。
>
> 传入参数的类型错误 (例如在要求 [`int`](https://docs.python.org/zh-cn/3/library/functions.html#int) 时却传入了 [`list`](https://docs.python.org/zh-cn/3/library/stdtypes.html#list)) 应当导致 [`TypeError`](https://docs.python.org/zh-cn/3/library/exceptions.html#TypeError)，但传入参数的值错误 (例如传入要求范围之外的数值) 则应当导致 [`ValueError`](https://docs.python.org/zh-cn/3/library/exceptions.html#ValueError)。



### `exception ValueError`

> 当操作或函数接收到具有正确类型但值不适合的参数，并且情况不能用更精确的异常例如 [`IndexError`](https://docs.python.org/zh-cn/3/library/exceptions.html#IndexError) 来描述时将被引发。



---

参考文章：[heapq --- 堆队列算法 — Python 3.12.0 文档](https://docs.python.org/zh-cn/3/library/heapq.html)

如有侵权行为，请告知删除。
