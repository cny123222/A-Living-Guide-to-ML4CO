---
title: 'Pytorch Tensors: A Beginner''s Guide'
date: 2025-08-16 13:12:38
index_img: img/ml4co.png
tags:
  - AI
category:
  - AI
sticky: 300
---

本文是一篇系统性的 Pytorch Tensor 入门教程，将介绍 Tensor 的基础及核心操作，包含了基础语法、索引操作、变形操作、计算操作等。其中高级索引和变形操作是理解 Tensor 操作的重难点。本文附有[项目代码（Jupyter Notebook）](https://github.com/cny123222/A-Living-Guide-to-ML4CO/blob/main/tensor.ipynb)。

<!-- more -->

## 引言

Tensor 是 Pytorch 基础的数据类型，其操作种类繁多，不易想象，但对理解代码逻辑至关重要。本文将从易至难，详细阐释其中较为常用的一些操作，帮助大家理解。

本文参考了密歇根大学 EECS 498-007 / 598-005: Deep Learning for Computer Vision 课程的 Assignment 1 [2] 中介绍 Pytorch 的部分。

## 基础语法

其实 Pytorch 的 Tensor 和 NumPy 类似，都是表示多维矩阵的，但是增加了 **GPU 加速和自动求梯度**的功能，专为深度学习打造。

### 创建操作

#### 与 NumPy 数组的转换

直接创建 Tensor 通常要**从 Python 列表或 NumPy 数组**转换。这里介绍较为常用的**与 NumPy 数组相互转换**的语法。

从 NumPy 数组到 Tensor 有两种方法：

- `torch.from_numpy()`：这种方法比较高效，因为避免了数据复制，而是使用内存共享。这意味着，修改其中一个，另一个会跟着改变。

```python
# 使用 `torch.from_numpy()` (共享内存)
a_array = np.array([1, 2, 3])
a_tensor = torch.from_numpy(a_array)
print("Original PyTorch tensor:", a_tensor)
a_array[0] = 10  # 修改 NumPy 数组
print("Tensor after modifying NumPy array:", a_tensor)
```
```plaintext
Original PyTorch tensor: tensor([1, 2, 3])
Tensor after modifying NumPy array: tensor([10,  2,  3])
```

- `torch.tensor()`：这种方法会创建一个新的数据副本，不共享内存。因此，存在数据复制开销，但是安全、独立。

```python
# 使用 `torch.tensor()` (复制内存)
b_array = np.array([[1, 2], [3, 4]])
b_tensor = torch.tensor(b_array)
print("Original PyTorch tensor:")
print(b_tensor)
b_array[0, 0] = 10  # 修改 NumPy 数组
print("Tensor after modifying NumPy array:")
print(b_tensor)
```
```plaintext
Original PyTorch tensor:
tensor([[1, 2],
        [3, 4]])
Tensor after modifying NumPy array:
tensor([[1, 2],
        [3, 4]])
```

我们顺便介绍从 Tensor 到 NumPy 数组的转换，一般使用 `.numpy()` 方法。但注意：
- **内存共享**：`.numpy()` 方法会使用共享内存，即指向内存中同一块地址。如果不希望它们相互影响，可以使用 `.clone().numpy()`。
- **必须在 CPU 上**：NumPy 数组是基于 CPU 内存的。如果我们的 Tensor 存储在 GPU 上，直接调用 `.numpy()` 会报错，必须先用 `.cpu()` 方法把它移回 CPU。

还要注意，如果一个 Tensor 需要计算梯度，那么在转换为 NumPy 数组之前，需要**先使用 `.detach()` 方法**，使得其脱离计算图。

```python
# 从 Tensor 转换为 NumPy 数组
c_tensor = torch.tensor([1, 2, 3], device='cuda')  # 创建一个在 GPU 上的 Tensor
try: # GPU 上的 Tensor 不能直接转换为 NumPy 数组
    c_array = c_tensor.numpy()
except Exception as e:
    print(e)
c_array = c_tensor.cpu().numpy()
print("NumPy array from CPU tensor:", c_array)
```
```plaintext
can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
NumPy array from CPU tensor: [1 2 3]
```

#### 特殊 Tensor 构造

我们常常需要**构造全 0、全 1、随机**的 Tensor。
- `torch.zeros()`：构造全 0 的 Tensor
- `torch.ones()`：构造全 1 的 Tensor
- `torch.rand()`：构造 [0,1] 均匀随机数的 Tensor

当然，还可以**由现有的 Tensor 进行类似构造**，如 `torch.zeros_like()` 可以创建和原 Tensor 形状相同的全 0 Tensor。

```python
a = torch.rand(2, 3)
print("Random tensor a:")
print(a)

b = torch.zeros_like(a)
print("\nTensor b with same shape as a:")
print(b)
```
```plaintext
Random tensor a:
tensor([[0.8838, 0.2225, 0.5416],
        [0.0396, 0.2708, 0.4012]])

Tensor b with same shape as a:
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

还有一些常用的构造函数。

- `torch.arange()`：类似 Python 的 `range`，返回一个 `[start, end)`，从 `start` 开始，步长为 `step` 的一维 Tensor。

```python
a = torch.arange(1, 6, 2)
print("Tensor a:", a)
```
```plaintext
Tensor a: tensor([1, 3, 5])
```

### 基本属性

- `.dim()`：返回 Tensor 的维度
- `.shape`：返回 Tensor 的形状
- `.shape[i]`：返回第 i 个维度的大小
- `.dtype`：返回数据类型
- `.device`：返回所在设备

```python
a = torch.zeros(2, 3, 4, device='cuda')
print("Rank of a:", a.dim())
print("Shape of a:", a.shape)
print("Shape of dim 1:", a.shape[1])
print("Datatype of a:", a.dtype)
print("Device of a:", a.device)
```
```plaintext
Rank of a: 3
Shape of a: torch.Size([2, 3, 4])
Shape of dim 1: 3
Datatype of a: torch.float32
Device of a: cuda:0
```

### 数据类型及设备

在创建 Tensor 时，可以用 `dtype` 参数**指定数据类型**，用 `device` 参数**指定所在设备**。一般默认的数据类型是 `torch.float32`，默认的设备是 `cpu`。

常用的数据类型有：
- `torch.float32`：标准的浮点类型，**网络参数默认采用**，是 Pytorch 中最常见的数据类型，可以使用 `.float()` 转换到该数据类型。
- `torch.int64`：通常用于存储索引，可以使用 `.long()` 转换到该数据类型。
- `torch.bool`：用于存储布尔值，可以使用 `.bool()` 转换到该数据类型。
- `torch.float16`：用于混合精度训练。

对于设备转换，可以用 `.cpu()` 和 `.cuda()` 在 CPU 和 GPU 之间搬运数据。

当然，**通用的转化方式是 `.to()`**，可以指定任意的数据类型或设备。

```python
x0 = torch.zeros(2, 3, dtype=torch.float16, device='cpu')
print("x0.dtype:", x0.dtype)
print("x0.device:", x0.device)
x1 = x0.float()
print("x1.dtype:", x1.dtype)
x2 = x0.long()
print("x2.dtype:", x2.dtype)
x3 = x0.to(torch.int32)
print("x3.dtype:", x3.dtype)
x4 = x0.to("cuda")
print("x4.device:", x4.device)
```
```plaintext
Datatype of x0: torch.float16
Device of x0: cpu
Datatype of x1: torch.float32
Datatype of x2: torch.int64
Datatype of x3: torch.int32
Device of x4: cuda:0
```

注意，Pytorch 中的创建和计算操作（如使用 `torch.zeros_like()`、`.add()` 等），得到的新 Tensor 都会**默认继承原有 Tensor 的数据类型和设备**。

## 索引操作

### 单元素索引

最简单的单元素索引大家都会，但注意直接索引返回的是 Pytorch 标量，要**调用 `.item()` 方法转换到 Python 标量**。单元素索引可以直接修改元素。

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("a[0, 1]:", a[0, 1])
print("a[0, 1].item():", a[0, 1].item())
```
```plaintext
a[0, 1]: tensor(2)
a[0, 1].item(): 2
```

### 切片索引

Tensor 和 Python 列表、NumPy 数组一样，有切片索引语法，语法也是 `start:stop` 或者 `start:stop:step`。注意：
- 索引包含 `start` 不包含 `stop`。
- `start` 和 `stop` 都可以是负数，表示从后向前的索引。
- `start` 省略表示从 0 开始，`stop` 省略表示到最后一个元素。

下面是几个简单但常用的例子：
```python
a = torch.tensor([10, 11, 12, 13, 14, 15, 16])
print("a[2:5] ", a[2:5])  # Elements between index 2 and 5
print("a[:-1] ", a[:-1])  # All elements except the last one
print("a[::2] ", a[::2])  # Every second element
print("a[:] ", a[:])      # All elements
```
```plaintext
a[2:5]  tensor([12, 13, 14])
a[:-1]  tensor([10, 11, 12, 13, 14, 15])
a[::2]  tensor([10, 12, 14, 16])
a[:]  tensor([10, 11, 12, 13, 14, 15, 16])
```

对于**多维 Tensor**，可以**在每个维度进行单元素或切片索引**。当某个维度进行单元素索引时，**该维度消失**；当某个维度进行切片索引时，该维度保持（即使该维度大小变为 1）。省略的后续维度默认全选。

```python
b = torch.tensor(
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]
)
# Single row
print("Single row:")
print(b[1, :], b[1, :].shape)  # Equivalent to b[1]
print(b[1:2, :], b[1:2, :].shape)
# Single column
print("\nSingle column:")
print(b[:, 2], b[:, 2].shape)
print(b[:, 2:3], b[:, 2:3].shape)
# All columns except the last one
print("\nAll columns except the last one:")
print(b[:, :-1], b[:, :-1].shape)
```
```plaintext
Single row:
tensor([5, 6, 7, 8]) torch.Size([4])
tensor([[5, 6, 7, 8]]) torch.Size([1, 4])

Single column:
tensor([ 3,  7, 11]) torch.Size([3])
tensor([[ 3],
        [ 7],
        [11]]) torch.Size([3, 1])

All columns except the last one:
tensor([[ 1,  2,  3],
        [ 5,  6,  7],
        [ 9, 10, 11]]) torch.Size([3, 3])
```

要注意的是，**切片使用内存共享，不会复制数据**。因此修改切片中的数据，原来 Tensor 中的数据也会改变。如果需要避免，使用 `.clone()` 方法。

**修改** Tensor 切片，可以直接用常数或新 Tensor 赋值。

```python
c = torch.zeros(2, 4, dtype=torch.int64)
c[:, :2] = 1
c[:, 2:] = torch.tensor([[2, 3], [4, 5]])
print(c)
```
```plaintext
tensor([[1, 1, 2, 3],
        [1, 1, 4, 5]])
```

### 整数 Tensor 索引

{% note info %}

**从这里开始，Tensor 真正奇妙的操作开始了！** 高级索引和变形是 Tensor 的核心操作，但并不好理解，我们尽量详细一些，配合一些例子。

{% endnote %}

切片索引有很强的局限性，其得到的 Tensor 只能是原 Tensor 的子矩阵。当我们引入**索引数组**，就会变得更灵活。

一切索引方式都**从单元素索引演化而来**。单元素索引的每个维度都用一个整数索引，表示该维度上选中某个特定的位置。当把一个维度上的整数换成一个切片 `start:stop:step`，就成为了切片索引，表示该维度上依次选择某些特定的位置。（对于处在最后的全部选择的维度，可以都省略。）

而当**把一个维度上的整数换成一个整数数组**时，就得到了整数 Tensor 索引。这个一维整数数组像一张购物清单，**表示了在这个维度上，要依次选择哪些索引**。我们举几个例子说明：

```python
a = torch.arange(12).reshape(3, 4)
print("Original tensor a:")
print(a)

idx = torch.tensor([0, 0, 2, 1, 1])
print('\nReordered rows:')
print(a[idx])

idx = torch.tensor([3, 2, 1, 0])
print('\nReordered columns:')
print(a[:, idx])
```
```plaintext
Original tensor a:
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

Reordered rows:
tensor([[ 0,  1,  2,  3],
        [ 0,  1,  2,  3],
        [ 8,  9, 10, 11],
        [ 4,  5,  6,  7],
        [ 4,  5,  6,  7]])

Reordered columns:
tensor([[ 3,  2,  1,  0],
        [ 7,  6,  5,  4],
        [11, 10,  9,  8]])
```

首先看 Reordered rows 的例子。第一个维度传入了一个列表，表示要按列表挑选；第二个维度默认全选，该维度将保留。在第一个维度上，Python 根据列表 `idx = tensor([0, 0, 2, 1, 1])` 的指示，依次取出第 0 行、第 0 行、第 2 行、第 1 行、第 1 行（第二个维度都是全选），并按该顺序在第一个维度上排列，形成结果中形状为 `(5, 4)` 的 Tensor。

再看 Reordered columns 的例子。第一个维度传入 `:`，表示全选，即对所有行都要执行操作。第二个维度传入列表 `idx = tensor([3, 2, 1, 0])`，表示该维度上需要挑选。Python 对于第一个维度中的每个对象（即每行），按第二个维度要求的索引列表进行挑选，即依此挑选该行的第 3 个、第 2 个、第 1 个、第 0 个元素，并按该顺序排列，形成结果中形状为 `(3, 4)` 的 Tensor。

当**索引中有多个维度传入一维列表（需长度相同）时**，Pytorch 会将这些列表配对，**在这几个维度上依此同时选中某一位置组合**。

我们从最简单的二维情况出发，来分析一个处理对角线元素的例子：
```python
b = torch.arange(1, 10).reshape(3, 3)
print("Original tensor b:")
print(b)

idx = torch.tensor([0, 1, 2])
print('\nGet the diagonal:')
print(b[idx, idx])

print('\nSet the diagonal to 0:')
b[idx, idx] = 0
print(b)
```
```plaintext
Original tensor b:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

Get the diagonal:
tensor([1, 5, 9])

Set the diagonal to 0:
tensor([[0, 2, 3],
        [4, 0, 6],
        [7, 8, 0]])
```

这时，索引的两个维度都传入了整数列表。Pytorch 对于传入列表（设长度均为 N）的维度，会依次同时取出各维度的索引列表的第 0 个、第 1 个、……、第 N-1 个元素，在各个维度上选中列表中对应元素指明的那一个位置。例如，`a[idx0, idx1]` 等价于：
```plaintext
torch.tensor([
  a[idx0[0], idx1[0]],
  a[idx0[1], idx1[1]],
  ...,
  a[idx0[N - 1], idx1[N - 1]]
])
```
这个例子中，由于两个维度的列表均为 `idx = tensor([0, 1, 2])`，这会依次选中 `b[0, 0]`、`b[1, 1]`、`b[2, 2]`，从而取出对角线元素。

再看一个从每行选出指定元素的例子：
```python
c = torch.arange(1, 13).reshape(4, 3)
print("Original tensor c:")
print(c)

idx0 = torch.arange(c.shape[0])
idx1 = torch.tensor([1, 2, 1, 0])
print('\nGet elements using index arrays:')
print(c[idx0, idx1])
```
```plaintext
Original tensor c:
tensor([[ 1,  2,  3],
        [ 4,  5,  6],
        [ 7,  8,  9],
        [10, 11, 12]])

Get elements using index arrays:
tensor([ 2,  6,  8, 10])
```
这里，`idx0 = tenser([0, 1, 2, 3])` 依次选中每一行，`idx1` 依次指定了第二个维度选中哪个位置，即每行取哪个元素。

可以这么理解：当整数列表索引和切片索引同时出现时，整数列表索引每次选中一个位置，切片索引每次选中整个切片。

### 布尔 Tensor 索引

布尔索引**可以更方便地筛选需要的元素**，通常会用于选择或修改 Tensor 中符合某些条件的元素。

```python
a = torch.rand(3, 4)
print("Original tensor:")
print(a)

mask = (a > 0.5)
print("\nMask tensor:")
print(mask)

print('\nSelecting elements with the mask:')
print(a[mask])

a[mask] = 0
print('\nAfter modifying with a mask:')
print(a)
```
```plaintext
Original tensor:
tensor([[0.3443, 0.5997, 0.8539, 0.3628],
        [0.1334, 0.7126, 0.5848, 0.5628],
        [0.7731, 0.5251, 0.0347, 0.6972]])

Mask tensor:
tensor([[False,  True,  True, False],
        [False,  True,  True,  True],
        [ True,  True, False,  True]])

Selecting elements with the mask:
tensor([0.5997, 0.8539, 0.7126, 0.5848, 0.5628, 0.7731, 0.5251, 0.6972])

After modifying with a mask:
tensor([[0.3443, 0.0000, 0.0000, 0.3628],
        [0.1334, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0347, 0.0000]])
```

这段代码展示并解释了布尔索引的常见用法，其作用是将 `a` 中所有大于 0.5 的元素修改为 0。

`a > 0.5` 会生成一个和 `a` 形状相同的 Tensor，数据类型为 `torch.bool`。其中大于 0.5 的元素变为 `True`，其余元素变为 `False`。对 `a` 使用这个 mask 进行索引，就能选出其中所有大于 0.5 的元素，并可以统一修改这些元素。

## 变形操作

变形操作也是 Tensor 核心而重要的操作。

### 改变逻辑形状

`.view()` 和 `.reshape()` 方法像是在重新解读一块连续的内存，它们可以**改变 Tensor 的形状，但不改变其中元素的排列顺序**。

`.view()` 和 `.reshape()` 基本语法相同，传入的参数是改变后的形状，返回一个具有新形状的张量。**可以有一个维度传入 `-1`**：由于改变形状不会使得元素数量变化，Pytorch 会自动推算这个维度的大小。

```python
x0 = torch.arange(1, 9).reshape(2, 4)
print('Original tensor:')
print(x0)
print('shape:', x0.shape)

x1 = x0.view(-1)  # Equivalent to x1 = x0.flatten()
print('\nFlattened tensor:')
print(x1)
print('shape:', x1.shape)

x2 = x1.reshape(-1, 1)
print('\nColumn vector:')
print(x2)
print('shape:', x2.shape)

x3 = x1.view(2, 2, 2)
print('\nRank 3 tensor:')
print(x3)
print('shape:', x3.shape)
```
```plaintext
Original tensor:
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
shape: torch.Size([2, 4])

Flattened tensor:
tensor([1, 2, 3, 4, 5, 6, 7, 8])
shape: torch.Size([8])

Column vector:
tensor([[1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8]])
shape: torch.Size([8, 1])

Rank 3 tensor:
tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
shape: torch.Size([2, 2, 2])
```

但是，这两个操作也有区别。`.view()` **保证不复制数据**，其创建的仅仅是一个视图 (View)，也就是说新张量和原张量**共享同一块内存数据**。修改其中一个，另一个会跟着改变。它**只适用于在内存中连续 (Contiguous) 的张量**。

相比之下，`.reshape()` **更强大、更安全**。它首先会尝试像 `.view()` 一样创建一个共享内存的视图。如果因为张量在内存中不连续而无法创建视图，它会**自动创建一个数据副本**，然后改变形状。**在实际应用中，应当优先考虑使用 `.reshape()`。**

```python
a = torch.arange(12).reshape(3, 4)
b = a.t() # .t() makes the tensor non-contiguous
print("Is b contiguous?", b.is_contiguous())
print("\nOriginal tensor b:")
print(b)

try:
    c = b.view(2, 6)
except Exception as e:
    print("\nview() Failed:", e)

d = b.reshape(2, 6)
print("\nreshape() Succeeded:")
print(d)
print("\nIs d contiguous?", d.is_contiguous())
```
```plaintext
Is b contiguous? False

Original tensor b:
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])

view() Failed: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

reshape() Succeeded:
tensor([[ 0,  4,  8,  1,  5,  9],
        [ 2,  6, 10,  3,  7, 11]])

Is d contiguous? True
```

在这里，我们看到 `.reshape()` 可以处理内存不连续的 Tensor，但 `.view()` 不可以。

### 改变维度顺序

这类操作不会改变每个维度的大小，而是**改变这些维度的位置**。常见的是 `.transpose()` 和 `.permute()`。其区别是，`.transpose(dim0, dim1)` **交换张量的某两个指定维度**，`.permute(*dims)` 能**按照指定顺序一次性重排所有维度**（一般用于三个及以上维度的换序）。这两个操作均创建一个视图，不复制数据。

```python
x0 = torch.arange(1, 25).reshape(2, 3, 4)
print('Original tensor:')
print(x0)
print('shape:', x0.shape)

x1 = x0.transpose(0, 1)
print('\nSwap axes 0 and 1:')
print(x1)
print(x1.shape)

x2 = x0.permute(1, 2, 0)
print('\nPermute axes')
print(x2)
print('shape:', x2.shape)
```
```plaintext
Original tensor:
tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],

        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]])
shape: torch.Size([2, 3, 4])

Swap axes 0 and 1:
tensor([[[ 1,  2,  3,  4],
         [13, 14, 15, 16]],

        [[ 5,  6,  7,  8],
         [17, 18, 19, 20]],

        [[ 9, 10, 11, 12],
         [21, 22, 23, 24]]])
torch.Size([3, 2, 4])

Permute axes
tensor([[[ 1, 13],
         [ 2, 14],
         [ 3, 15],
         [ 4, 16]],

        [[ 5, 17],
         [ 6, 18],
         [ 7, 19],
         [ 8, 20]],

        [[ 9, 21],
         [10, 22],
         [11, 23],
         [12, 24]]])
shape: torch.Size([3, 4, 2])
```

但是，换维度顺序这件事比较难以理解。很多人大概只能理解到矩阵转置，即二维 Tensor 交换维度顺序的操作。一种理解是，这两个操作不会移动任何元素的位置，而是改变元素的查找和组织方式，也就是查找这个元素所用索引的顺序。例如，当对 `a` 做 `permute(1, 2, 0)` 后，原先用 `a[i, j, k]` 能找到的元素，现在可以用 `a[j, k, i]` 找到，即我们只需要改变索引的顺序。总的来说，这两个操作**只改变了 Tensor 如何解读自己的数据，而没有进行任何数据的复制或移动**。

注意，**改变维度后的 Tensor 在内存中不连续，可以调用 `.contiguous()` 方法使它们连续**，比如紧接着要做 `.view()` 操作等。

## 计算操作

### 逐元素操作

这部分比较简单，无非是加减乘除、乘方、开方、三角函数，和 NumPy 语法几乎一致。注意 **`x * y` 是逐元素乘法**就可以，而不是矩阵乘法。以加法为例，`torch.add(x, y)`、`x.add(y)`、`x + y` 效果完全一样，其余运算类似。

### 归约操作

归约操作是指对 Tensor 的某一部分汇总来计算某种数值的操作。常见的有 `.sum()`、 `.min()`、 `.max()`、 `.mean()`、 `.argmax()` 等，语法基本一致。这里以 `.sum()` 为例。

```python
a = torch.arange(1, 25).reshape(2, 3, 4)
print("Original tensor:")
print(a, a.shape)

print('\nSum over entire tensor:')
print(a.sum(), a.sum().shape)

print('\nSum over the first dimension:')
print(a.sum(dim=0), a.sum(dim=0).shape)

print('\nSum over the second dimension:')
print(a.sum(dim=1), a.sum(dim=1).shape)

print('\nSum over the last dimension:')
print(a.sum(dim=-1), a.sum(dim=-1).shape)
```
```plaintext
Original tensor:
tensor([[[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12]],

        [[13, 14, 15, 16],
         [17, 18, 19, 20],
         [21, 22, 23, 24]]]) torch.Size([2, 3, 4])

Sum over entire tensor:
tensor(300) torch.Size([])

Sum over the first dimension:
tensor([[14, 16, 18, 20],
        [22, 24, 26, 28],
        [30, 32, 34, 36]]) torch.Size([3, 4])

Sum over the second dimension:
tensor([[15, 18, 21, 24],
        [51, 54, 57, 60]]) torch.Size([2, 4])

Sum over the last dimension:
tensor([[10, 26, 42],
        [58, 74, 90]]) torch.Size([2, 3])
```

不传递参数时，返回所有元素的和（Pytorch 标量）。传递 `dim` 参数时，按指定的维度计算元素和，返回的 Tensor 中该维度消失，其余维度不变。若要保留求和维度，可以传递 `keepdim=True`。

### 矩阵操作

Tensor 的矩阵操作也比较简单，建议**直接用 `@` 或 `torch.matmul()` 进行矩阵乘法（或向量）**。它会自动根据输入维度，选择进行矩阵乘法、矩阵乘向量、向量内积、带批次维度的矩阵乘法等。

### 广播机制

Pytorch 的 Tensor 与 NumPy 数组类似，也有广播机制。最常见的是**逐元素操作和矩阵乘法的广播机制**。广播机制使得在不实际复制数据的情况下，让 PyTorch 表现得好像它扩展了较小张量的形状，以匹配较大张量的形状。

使用广播机制需要**检查维度是否匹配**。Pytorch 会先在较低维度 Tensor 的维度前补 1，直到两个 Tensor 维度相等。接着依次检查每个维度的大小是否相等或其中一个为 1。若检查通过，则可以进行广播。**某个维度大小为 1 的 Tensor 会表现得和其沿着该维度复制了一样。**

接下来举几个例子：

1. **计算向量的外积（张量积）**
$$
\mathbf{a} \otimes \mathbf{b} = \begin{bmatrix}a_1 \\ a_2 \\ a_3\end{bmatrix} \otimes \begin{bmatrix} b_1 & b_2 \end{bmatrix} = \begin{bmatrix} a_1b_1 & a_1b_2 \\ a_2b_1 & a_2b_2 \\ a_3b_1 & a_3b_2\end{bmatrix}
$$

```python
# Compute outer product of vectors
v1 = torch.tensor([1, 2, 3])  # shape: (3,)
v2 = torch.tensor([4, 5])     # shape: (2,)
print(v1.reshape(-1, 1) * v2)
```
```plaintext
tensor([[ 4,  5],
        [ 8, 10],
        [12, 15]])
```

`v1.view(-1, 1)` 的形状为 `(3, 1)`，可以和对 `v2` 广播后相乘，得到的矩阵形状为 `(3, 2)`，正好是 `v1` 和 `v2` 的外积结果。

2. 对矩阵的每一行加一个向量

```python
# Add the vector to each row of the matrix
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
v = torch.tensor([1, 2, 3])               # shape: (3,)
print('Here is the matrix:')
print(x)
print('\nHere is the vector:')
print(v)
print('\nAdd the vector to each row of the matrix:')
print(x + v)
```
```plaintext
Here is the matrix:
tensor([[1, 2, 3],
        [4, 5, 6]])

Here is the vector:
tensor([1, 2, 3])

Add the vector to each row of the matrix:
tensor([[2, 4, 6],
        [5, 7, 9]])
```

`x` 的形状为 `(2, 3)`，`v` 的形状为 `(3,)`。它们会被广播到 `(2, 3)`，即 `v` 会复制给 `x` 的每一行，然后相加，故 `x` 的每一行都加上了 `v`。

3. 对矩阵的每一列加一个向量

```python
# Add the vector to each column of the matrix
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
w = torch.tensor([4, 5])                  # shape: (2,)
print('Here is the matrix:')
print(x)
print('\nHere is the vector:')
print(w)
print('\nAdd the vector to each column of the matrix:')
print(x + w.reshape(-1, 1))
```
```plaintext
Here is the matrix:
tensor([[1, 2, 3],
        [4, 5, 6]])

Here is the vector:
tensor([4, 5])

Add the vector to each column of the matrix:
tensor([[ 5,  6,  7],
        [ 9, 10, 11]])
```

这与刚才类似，不同的是 `w.view(-1, 1)` 的形状为 `(2, 1)`，广播机制使得其在第二个维度上复制，形状变为 `(2, 3)` 后与 `v` 相加。

4. 对一个矩阵乘一系列常数

```python
# Multiply a tensor by a set of constants
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
c = torch.tensor([1, 10, 11, 100])        # shape: (4,)
print('Here is the matrix:')
print(x)
print('\nHere is the vector:')
print(c)
print('\nMultiply x by a set of constants:')
print(c.reshape(-1, 1, 1) * x)
```
```plaintext
Here is the matrix:
tensor([[1, 2, 3],
        [4, 5, 6]])

Here is the vector:
tensor([  1,  10,  11, 100])

Multiply x by a set of constants:
tensor([[[  1,   2,   3],
         [  4,   5,   6]],

        [[ 10,  20,  30],
         [ 40,  50,  60]],

        [[ 11,  22,  33],
         [ 44,  55,  66]],

        [[100, 200, 300],
         [400, 500, 600]]])
```

`x` 的初始维度为 `(2, 3)`，`c` 的初始维度为 `(4, )`。这 4 个常数将分别作用于整个矩阵，`4` 会最终出现在第一个维度上，因此先将 `c` 的形状变为 `(4, 1, 1)`，再进行广播。`c` 中每个数会对 `x` 中每个元素复制，`x` 会对每个 `c` 中的数复制，最终形状变为 `(4, 2, 3)`。

## 参考资料
[1] [https://docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)
[2] [https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment1.html](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment1.html)