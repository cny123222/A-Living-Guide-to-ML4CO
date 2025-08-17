---
title: Fancy but Useful Tensor Operations
date: 2025-08-14 23:38:14
index_img: img/ml4co.png
tags:
  - AI
category:
  - AI
sticky: 300
---

代码中看似花哨的 Tensor 操作常常让人摸不着头脑，但是恰恰是理解深度学习代码的关键。本文将详细介绍一些**常见、有趣且实用的 Tensor 操作**，项目代码（Jupyter Notebook）在[这里](https://github.com/cny123222/A-Living-Guide-to-ML4CO/blob/main/tensor2.ipynb)。

<!-- more -->

## 引言

笔者之前写过一稿这篇博客，但逐渐写成了 Pytorch Tensor 教程，于是将其放在了 [Pytorch Tensors: A Beginner's Guide](https://cny123222.github.io/2025/08/16/Pytorch-Tensors-A-Beginner-s-Guide/) 这里，其中**系统介绍了 Tensor 的基础操作**，包括高级索引、变形等比较重点的操作。**这篇博客是收集向的**，将会逐步收集我看代码（特别是 ML4CO 代码）的时候碰到的有趣且有用的 Tensor 操作，还有一些之前的博客没放进去的常用操作。

## 增减维度：squeeze & unsqueeze

在 PyTorch 中，我们经常需要调整张量的形状以适应不同函数或神经网络层的输入要求。其中`.squeeze()` 和 `.unsqueeze()` 是**用于增减维度**的常用操作，它们分别用于**添加新维度**和**移除多余维度**。

`tensor.unsqueeze(dim)` 用于**在指定的维度上增加一个大小为 1 的新维度**。参数 `dim` 指定了新维度的插入位置。该操作返回一个视图 (View)，共享内存。

```python
x0 = torch.tensor([1, 2, 3])
print("Original tensor:")
print(x0, x0.shape)

# Add a new dimension at position 0
x1 = x0.unsqueeze(0)
print("\nAdd a new dimension at position 0:")
print(x1, x1.shape)

# Add a new dimension at position 1
x2 = x0.unsqueeze(1)
print("\nAdd a new dimension at position 1:")
print(x2, x2.shape)
```
```plaintext
Original tensor:
tensor([1, 2, 3]) torch.Size([3])

Add a new dimension at position 0:
tensor([[1, 2, 3]]) torch.Size([1, 3])

Add a new dimension at position 1:
tensor([[1],
        [2],
        [3]]) torch.Size([3, 1])
```

`tensor.squeeze(dim=None)` 用于**移除 Tensor 中大小为 1 的维度**。若不提供 `dim` 参数，会移除所有大小为 1 的维度；若提供 `dim` 参数，它只会检查并移除指定的维度（如果大小为 1）。该操作返回一个视图 (View)，共享内存。

```python
y0 = torch.rand(1, 3, 1, 5)
print("Original tensor shape:", y0.shape)

# Remove all dimensions of size 1
y1 = y0.squeeze()
print("\nShape after removing all dimensions of size 1:", y1.shape)

# Remove dimension at position 0
y2 = y0.squeeze(0)
print("\nShape after removing dim=0:", y2.shape)
```
```plaintext
Original tensor shape: torch.Size([1, 3, 1, 5])

Shape after removing all dimensions of size 1: torch.Size([3, 5])

Shape after removing dim=0: torch.Size([3, 1, 5])
```

## 合并张量：cat & stack

当我们需要**将多个张量合并成一个**时，PyTorch 提供了两个核心函数：`torch.cat()` (concatenate) 和 `torch.stack()`。

`torch.cat(tensors, dim=0)` 用于将一系列 Tensor **沿着一个已经存在的维度**进行连接，其中 `tensors` 是 Tensor 的列表或元组，`dim` 指定沿着哪个维度进行拼接。除了被连接的维度可以不同，**其他维度的大小必须保持一致**。注意，`cat()` 操作**不会增加 Tensor 的维度**。

```python
t1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
t2 = torch.tensor([[7, 8, 9],
                   [10, 11, 12]])
print("Original tensors:")
print(t1, t1.shape)
print(t2, t2.shape)

# Concatenate t1 and t2 along dimension 0
t3 = torch.cat((t1, t2), dim=0)
print("\nConcatenated on dim=0:")
print(t3, t3.shape)

# Concatenate t1 and t2 along dimension 1
t4 = torch.cat((t1, t2), dim=1)
print("\nConcatenated on dim=1:")
print(t4, t4.shape)
```
```plaintext
Original tensors:
tensor([[1, 2, 3],
        [4, 5, 6]]) torch.Size([2, 3])
tensor([[ 7,  8,  9],
        [10, 11, 12]]) torch.Size([2, 3])

Stacked on dim=0:
tensor([[[ 1,  2,  3],
         [ 4,  5,  6]],

        [[ 7,  8,  9],
         [10, 11, 12]]]) torch.Size([2, 2, 3])

Stacked on dim=1:
tensor([[[ 1,  2,  3],
         [ 7,  8,  9]],

        [[ 4,  5,  6],
         [10, 11, 12]]]) torch.Size([2, 2, 3])
```

注意到，两个形状 `(2, 3)` 的 Tensor，在 `dim=0` 上拼接，第 0 维大小相加，其余维度大小不变，得到形状 `(4, 3)` 的 Tensor；在 `dim=1` 上拼接，第 1 维大小相加，其余维度大小不变，得到形状 `(2, 6)` 的 Tensor。

`torch.stack(tensors, dim=0)` 用于将一系列 Tensor **沿着一个全新的维度**进行堆叠，其中 `tensors` 是 Tensor 的列表或元组，`dim` 指定新维度插入的位置。`stack()` 要求**所有待堆叠的张量形状必须完全相同**。注意，`stack()` 操作**会增加一个新的维度**。

```python
t1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
t2 = torch.tensor([[10, 11, 12],
                   [13, 14, 15],
                   [16, 17, 18]])
print("Original tensors:")
print(t1, t1.shape)
print(t2, t2.shape)

# Stack t1 and t2 along dimension 0
t3 = torch.stack((t1, t2), dim=0)
print("\nStacked on dim=0:")
print(t3, t3.shape)

# Stack t1 and t2 along dimension 1
t4 = torch.stack((t1, t2), dim=1)
print("\nStacked on dim=1:")
print(t4, t4.shape)
```
```plaintext
Original tensors:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]) torch.Size([3, 3])
tensor([[10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]]) torch.Size([3, 3])

Stacked on dim=0:
tensor([[[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9]],

        [[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]]]) torch.Size([2, 3, 3])

Stacked on dim=1:
tensor([[[ 1,  2,  3],
         [10, 11, 12]],

        [[ 4,  5,  6],
         [13, 14, 15]],

        [[ 7,  8,  9],
         [16, 17, 18]]]) torch.Size([3, 2, 3])
```

注意到，两个形状 `(3, 3)` 的 Tensor，在 `dim=0` 上堆叠，会在第 0 维插入一个大小为 2 的维度，其余维度不变，得到形状 `(2, 3, 3)` 的 Tensor；在 `dim=1` 上堆叠，会在第 1 维插入一个大小为 2 的维度，其余维度不变，得到形状 `(3, 2, 3)` 的 Tensor。

## 扩展张量：expand & repeat

我们经常需要**将一个小 Tensor 复制多次**以匹配另一个大 Tensor 的形状，从而进行逐元素计算。PyTorch 提供了两种主要方法来实现这一目标：`expand` 和 `repeat`。它们得到的结果类似，但工作原理和性能开销不同。

`tensor.expand(*sizes)` **不会分配新内存来存储数据**，而是创建一个视图（View），即输出 Tensor 和原始 Tensor 共享内存。它**只能扩展大小为 1 的维度**，不能扩展大小大于 1 的维度。在目标形状参数中，`-1` 表示保持该维度的大小不变。

```python
x0 = torch.tensor([[1], [2], [3]]) # shape: (3, 1)
print("Original tensor:")
print(x0, x0.shape)

# Expand the dimension of size 1 to size 4.
x1 = x0.expand(3, 4)
print("\nExpanded tensor:")
print(x1, x1.shape)

# Modify an element in the original tensor
x0[0][0] = 100
print("\nExpanded tensor after modifying original tensor:")
print(x1, x1.shape)
```
```plaintext
Original tensor:
tensor([[1],
        [2],
        [3]]) torch.Size([3, 1])

Expanded tensor:
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]]) torch.Size([3, 4])

Expanded tensor after modifying original tensor:
tensor([[100, 100, 100, 100],
        [  2,   2,   2,   2],
        [  3,   3,   3,   3]]) torch.Size([3, 4])
```

`tensor.repeat(*repeats)` **会分配新的内存**，即新的 Tensor 拥有自己独立的数据存储。它的参数是重复的次数，而不是目标形状。它可以重复任何大小的维度，不局限于大小为 1 的维度。

```python
x0 = torch.tensor([[1], [2], [3]]) # shape: (3, 1)
print("Original tensor:")
print(x0, x0.shape)

# Repeat the tensor 3 times along dimension 1
x1 = x0.repeat(1, 3)
print("\nRepeated tensor:")
print(x1, x1.shape)

# Modify an element in the original tensor
x0[0][0] = 100
print("\nRepeated tensor after modifying original tensor:")
print(x1, x1.shape)
```
```plaintext
Original tensor:
tensor([[1],
        [2],
        [3]]) torch.Size([3, 1])

Repeated tensor:
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]]) torch.Size([3, 3])

Repeated tensor after modifying original tensor:
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]]) torch.Size([3, 3])
```

## 高级索引：gather & scatter_

`gather()` 和 `scatter_()` 属于高级索引操作，它们能让我们**根据一个索引张量（index）来并行地、选择性地从源张量（input 或 src）中取值或放值**。

首先，形象地理解一下这两个操作：

`torch.gather` 好比是“**按图索骥**”。想象一个巨大的仓库（`input`），里面有很多排货架。你手上有一张购物清单（`index`），清单上的每一项都精确地告诉你“在第几排的第几个位置取一个货物”。`gather()` 根据这张清单，并行地把所有需要的货物取出来，组成一个新的张量。

`tensor.scatter_()` 好比是“**对号入座**”。现在反过来，你有一批新货物（`src`）和一张入库单（`index`）。入库单告诉你应该把这批货物分别放到仓库的哪些指定位置上。`scatter_()` 将你手中的货物精准地“散布”到仓库的指定位置，从而更新仓库里的货物。

接着，我们重新结合例子严谨地解释一下：

`torch.gather(input, dim, index)` 的作用是：**沿着指定的维度 `dim`，根据 `index` 张量中的值，从 `input` 张量中选取元素**。注意，输出张量 `output` 的形状与 `index` 张量的形状完全相同；`input` 和 `index` 的维度数量必须相同；除了要操作的维度 `dim` 之外，`input` 和 `index` 在其他维度上的大小必须一致。

我们以强化学习中的 Q 值查询为例。假设我们有一个模型，它对一个批次（`batch_size=2`）中的每个状态，都预测了 4 个可能动作的 Q 值。我们想从 `q_values` 中抽取出我们实际采取的动作对应的 Q 值。这时就轮到 `gather()` 出场了。注意，我们在 action 维度（即 `dim=1`）上进行索引。

```python
# Assume batch_size=2, num_actions=4
# Each row represents a state, and each column represents the Q-value for an action.
q_values = torch.tensor([[0.1, 0.5, 0.2, 0.2],  # Q-values for state 1
                         [0.8, 0.1, 0.05, 0.05]]) # Q-values for state 2

# Assume these are the actions we actually took for each state
# (action 1 for the first state, action 0 for the second state).
# Note the shape is (2, 1) to match the dimensions needed for gather.
actions = torch.tensor([[1], [0]])

# dim=1 means we are indexing along the action dimension.
# index==actions tells gather which column to pick for each row.
# For row 0, it will pick the element at index 1 (which is 0.5).
# For row 1, it will pick the element at index 0 (which is 0.8).
selected_q_values = torch.gather(q_values, dim=1, index=actions)

print("Q-Values:")
print(q_values, q_values.shape)
print("Actions:")
print(actions, actions.shape)
print("\nGathered Q-Values:")
print(selected_q_values, selected_q_values.shape)
```
```plaintext
Q-Values:
tensor([[0.1000, 0.5000, 0.2000, 0.2000],
        [0.8000, 0.1000, 0.0500, 0.0500]]) torch.Size([2, 4])
Actions:
tensor([[1],
        [0]]) torch.Size([2, 1])

Gathered Q-Values:
tensor([[0.5000],
        [0.8000]]) torch.Size([2, 1])
```

`tensor.scatter_(dim, index, src)`（`src` 是张量）或 `tensor.scatter_(dim, index, value)`（`value` 是标量） 是 `gather()` 的逆操作。它使用 `index` 张量来定位，将源（`src` 张量或 `value` 标量）中的数据写入到原张量中。注意，**函数名末尾的下划线 `_` 表示这是一个 in-place 操作**，即它会直接修改调用它的那个张量。PyTorch 也提供了一个非 in-place 的 scatter 版本，但 scatter_ 在实践中更为普遍。注意，`index` 张量的维度数量必须和原张量的维度数量相同；如果使用 `src` 张量，`src` 的形状必须和 `index` 的形状完全相同。

我们以类别标签转换为 One-Hot 编码为例。

```python
# Assume batch_size=4, num_classes=5
# Class labels
labels = torch.tensor([1, 4, 2, 0])

# 1. Create a base tensor of zeros with shape (batch_size, num_classes)
one_hot = torch.zeros(4, 5)

# 2. Prepare the index and src arguments
# The index tensor needs to have the same number of dimensions as the
# one_hot tensor, so we use unsqueeze to add a dimension.
# labels.unsqueeze(1) gives it the shape (4, 1).
labels = labels.unsqueeze(1)
value = 1.0 # The value we want to fill in

# 3. Use scatter_ to perform the one-hot encoding
# dim=1 means we will be scattering values along the columns,
# at the positions specified by 'index'.
one_hot.scatter_(dim=1, index=labels, value=value)

print("Labels:")
print(labels, labels.shape)
print("\nOne-Hot Encoding:")
print(one_hot, one_hot.shape)
```
```plaintext
Labels:
tensor([[1],
        [4],
        [2],
        [0]]) torch.Size([4, 1])

One-Hot Encoding:
tensor([[0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 0.]]) torch.Size([4, 5])
```

## 持续更新中...

## 参考资料
[1] [https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)