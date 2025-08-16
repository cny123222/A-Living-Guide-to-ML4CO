---
title: Understading GNN - An ML4CO perspective
date: 2025-07-26 22:13:14
index_img: img/ml4co.png
tags:
  - AI
  - ML4CO
category:
  - ML4CO
  - Basics
sticky: 300
---

本期讲解 ML4CO 中的常见网络结构 GNN/GCN，并附有代码示例。项目代码可以在 [https://github.com/cny123222/A-Living-Guide-to-ML4CO](https://github.com/cny123222/A-Living-Guide-to-ML4CO) 中找到。

<!-- more -->

## 引言

**图神经网络**（GNN）常在 ML4CO 中被用作骨干网络。一方面，TSP 等经典 CO 问题是以图为背景的，GNN 能更好反映图结构；另一方面，GNN 对图的编码与顶点次序无关，因此 GNN 被 ML4CO 广泛使用。[1]

本文将介绍**图卷积神经网络（GCN）的基础知识**，以及**如何在 ML4CO 实践中用 Pytorch 搭建基于 GCN 的简单模型**。

## GCN 概述

众所周知，图由顶点（node）和边（edge）组成。在 GNN 中，我们为每个顶点和每条边赋予一个嵌入向量（embedding），来表征顶点或边的特征，并在每层中更新这一特征向量。

以 TSP 问题为例，我们的网络由 **Embedding 层、图卷积层、输出层**组成。输入可以是顶点和边的某些特征，输出一般是一张 heatmap，用来表示每条边出现在最短回路中的概率。

接下来，我们将进行逐层搭建。

{% note warning %}
注意，这里只使用**最简单的模型**进行演示。完整模型的训练及测试见 [A Living Guide to ML4CO](https://cny123222.github.io/2025/07/25/A-Living-Guide-to-ML4CO) 的“两种经典范式”部分。效果更好的经典模型及方法见“ML4CO 论文精读”系列。
{% endnote %}

## Embedding 层及其搭建

### 模型结构

假设输入是顶点的二维坐标 $x_i \in [0, 1]^2$，我们可以通过**线性变换**将其变成 $h$ 维的顶点嵌入：
$$
\alpha_i = A_1 x_i + b_1
$$
其中 $A_1 \in \mathbb{R}^{h\times2}$。对于边 $e_{ij}$，在这里我们简单地**用距离 $d_{ij}$ 作为特征**进行嵌入：
$$
\beta_i = A_2 d_{ij} + b_2
$$
其中 $A_2 \in \mathbb{R}^{h\times1}$。

### 代码搭建

接下来，我们用 Pytorch 编写 Embedding 层的简单代码。

```python
# gnn/embedder.py
from torch import Tensor, nn

class Embedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Embedder, self).__init__()
        self.node_embed = nn.Linear(2, hidden_dim, bias=True)
        self.edge_embed = nn.Linear(1, hidden_dim, bias=True)
        
    def forward(self, x: Tensor, e: Tensor):
        """
        Args:
            x: (V, 2) nodes_feature (node coords)
            e: (E,) edges_feature (distance matrix)
        Return:
            x: (V, H)
            e: (E, H)
        """  
        e = e.unsqueeze(-1)  # shape: (E, 1)
        x = self.node_embed(x) 
        e = self.edge_embed(e)
        return x, e
```

注意：
- 这里边的特征表示为一个 `E` 维向量，其中每个元素是边的长度。
- `e = e.unsqueeze(-1)` 的作用是给 `e` 在最后扩展一个新维度，从而能输入 Embedding 层。

## 图卷积层及其搭建

### 模型结构

在第 $l$ 层图卷积层中，输入的顶点特征 $x_i^l$ 和边特征 $e_i^l$ 进行传递和更新，输出更新后的 $x_i^{l+1}$ 和边特征 $e_i^{l+1}$。其中，第 0 层的输入为嵌入向量，即 $x_i^{0} = \alpha_i$，$e_i^{0} = \beta_i$。

为简单起见，我们用如下规则更新特征向量。对于边 $x_i$，使用与其相邻的顶点和边的特征来更新：
$$
x_i^{l+1} = x_i^l + \mathrm{ReLU}(\mathrm{LN}(W_1^l x_i^l + \sum_{j \sim i} e_{ij}^l \odot W_2^l x_j^l))
$$
其中 $W_1, W_2 \in \mathbb{R}^{h \times h}$，$\mathrm{LN}$ 是 Layer Normalization，$\odot$ 表示逐元素相乘。

而对于边 $e_{ij}$，也使用其相邻的顶点的特征来更新：
$$
e_{ij}^{l+1} = e_{ij}^l + \mathrm{ReLU}(\mathrm{LN}(W_3^l e_{ij}^l + W_4^l x_i^l + W_5^l x_j^l))
$$
其中 $W_3, W_4, W_5 \in \mathbb{R}^{h \times h}$。

### 代码搭建

这里，我们用代码搭建单层 GCN 层。

```python
# gnn/gnn_layer.py
from torch import Tensor, nn
import torch
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GNNLayer, self).__init__()
        
        # node updates
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # edge updates
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # LayerNorm for node and edge
        self.bn_x = nn.LayerNorm(hidden_dim)
        self.bn_e = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor):
        """
        Args:
            x: (V, H) Node features; e: (E, H) Edge features
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            Updated x and e after one layer of GNN.
        """
        # Deconstruct edge_index
        src, dest = edge_index  # shape: (E, )
        
        # --- Node Update ---
        w2_x_src = self.W2(x[src])  # shape: (E, H)
        messages = e * w2_x_src   # shape: (E, H)
        aggr_messages = torch.zeros_like(x)   # shape: (V, H)
        # index_add_ adds the 'messages' to 'aggr_messages' at indices specified by 'dest'
        aggr_messages.index_add_(0, dest, messages)   # shape: (V, H)
        x_new = x + F.relu(self.bn_x(self.W1(x) + aggr_messages))   # shape: (V, H)
        
        # --- Edge Update ---        
        w3_e = self.W3(e)  # shape: (E, H)
        w4_x_dest = self.W4(x[dest])  # shape: (E, H)
        w5_x_src = self.W5(x[src])  # shape: (E, H)
        
        e_new = e + F.relu(self.bn_e(w3_e + w4_x_dest + w5_x_src))   # shape: (E, H)

        return x_new, e_new
```

{% note info %}

这里涉及到一些 **Tensor 操作**。如果你不熟悉，可以翻看我的博客 [Pytorch Tensors: A Beginner's Guide](https://cny123222.github.io/2025/08/16/Pytorch-Tensors-A-Beginner-s-Guide/)，里面介绍了包括高级索引在内的 Tensor 核心操作。

{% endnote %}

这段代码中，顶点特征更新的代码不好理解，让我们逐句分析一下。

```python
src, dest = edge_index  # shape: (E, )
```

这句提取出了边的源顶点列表 `src` 和目的顶点列表 `dest`。它们都是 `E` 维向量，第 `i` 条边的源顶点索引为 `src[i]`，目的顶点索引为 `dest[i]`。

```python
w2_x_src = self.W2(x[src])  # shape: (E, H)
```

`x[src]` 是 Tensor 的高级索引操作。`x` 的第 `i` 行是第 `i` 个节点的特征向量。`x[src]` 从顶点特征表 `x` 中找出 `src` 中每个顶点的特征，并汇聚成一个形状为 `(E, H)` 的张量，其中第 `i` 行是第 `i` 条边的源顶点的特征向量。然后对每个顶点的特征应用 `self.W2` 的线性变换，得到 `w2_x_src`。

```python
messages = e * w2_x_src   # shape: (E, H)
```

这一步将每条边的特征向量及其源顶点的特征向量进行**逐元素相乘**，最终得到的 `messages` 有 `E` 行，第 `i` 行是融合了第 `i` 条边及其源顶点特征的特征向量。

```python
aggr_messages = torch.zeros_like(x)   # shape: (V, H)
aggr_messages.index_add_(0, dest, messages)   # shape: (V, H)
```

这两步比较关键，`index_add_` 函数（**下划线表示原地操作**）的作用是**按索引添加**。
- `dim=0`: 表示沿着第 0 维（节点的维度）进行添加；
- `index=dest`: 表示按照 `dest` 中的每一个顶点索引来分发 `messages`；
- `source=messages`: 表示要添加的内容是 `messages`。

这个操作会同时遍历 `messages` 的每一行和 `dest` 中的索引，并将该 message 加到 `aggr_messages` 中对应索引的行上（这一行即对应这条边的目的顶点）。

这样，`aggr_messages` 中的第 `i` 行就收到了所有以它为目的顶点的边的 message。

{% fold info @举一个简单的例子： %}

假设我们有 3 个节点 (V=3) 和 4 条边 (E=4)，

- `edge_index` = `[[0, 0, 1, 2], [1, 2, 2, 0]]`，
- `messages` 是一个 `4xH` 的张量，我们记为 `[m0, m1, m2, m3]`，
- `aggr_messages` 是一个 `3xH` 的全零张量。

那么，`index_add_` 的执行过程如下：
- 读取第一条消息 `m0`，它的目标是 `dest[0]=1`。于是 `aggr_messages[1] += m0`。
- 读取第二条消息 `m1`，它的目标是 `dest[1]=2`。于是 `aggr_messages[2] += m1`。
- 读取第三条消息 `m2`，它的目标是 `dest[2]=2`。于是 `aggr_messages[2] += m2`。
- 读取第四条消息 `m3`，它的目标是 `dest[3]=0`。于是 `aggr_messages[0] += m3`。

执行完毕后：
- `aggr_messages` 的第 0 行 = `m3` (节点 0 收到的消息之和)
- `aggr_messages` 的第 1 行 = `m0` (节点 1 收到的消息之和)
- `aggr_messages` 的第 2 行 = `m1 + m2` (节点 2 收到的消息之和)

{% endfold %}

其余代码比较容易理解，在此不再赘述。

## 输出层及其搭建

### 模型结构

在输出层中，我们需要**输出每条边出现在最终 tour 中的概率**，因此需要使用**最后一个 GCN 层的边特征向量**。我们考虑使用**多层感知机**（MLP）给出最后的概率 $p_{ij}^{\mathrm{TSP}}$：
$$
p_{ij}^{\mathrm{TSP}} = \mathrm{MLP}(e_{ij}^L)
$$
其中 MLP 的层数可以调整。

注意，这里我们并没有将输出结果转换为邻接矩阵的形式。当前输出的 `E` 维向量已经可以计算损失，邻接矩阵只会在 decoding 的过程中用到，我们之后进行实现。

### 代码搭建

输出层的实现较为简单，只包含多层感知机。

```python
# gnn/out_layer.py
from torch import Tensor, nn

class OutLayer(nn.Module):
    def __init__(self, hidden_dim: int, layer_num: int):
        """
        Args:
            hidden_dim: The dimension of the input edge features.
            layer_num: The number of layers in the MLP.
        """
        super(OutLayer, self).__init__()
        mlp_layers = []
        if layer_num == 1:
            mlp_layers.append(nn.Linear(hidden_dim, 2))
        else:
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            for _ in range(layer_num - 2):
                mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, e_final: Tensor):
        """     
        Args:
            e_final: (E, H) Final edge features
        Returns:
            prob: (E, 2) Probability of each edge being connected and not connected to the TSP tour.
        """
        prob = self.mlp(e_final)  # shape: (E, 2)
        return prob
```

注意，最后输出的预测向量维度是 `(E, 2)`，分别表示每条边“在”和“不在”最终 tour 中的概率。但这里我们**还没有做 softmax 归一化**，因为在计算 loss 时会包含 softmax 过程。

## Encoder 整体实现

在这里，我们对整个 Encoder 网络进行搭建。它包含了一个 Embedding 层、多个 GCN 层和一个输出层。

```python
# gnn/encoder.py
import torch
from torch import Tensor, nn
from .embedder import Embedder
from .gnn_layer import GNNLayer
from .out_layer import OutLayer
import torch.nn.functional as F

class GCNEncoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        gcn_layer_num: int,
        out_layer_num: int,
    ):
        super(GCNEncoder, self).__init__()
        self.embed = Embedder(hidden_dim)
        self.gcn_layers = nn.ModuleList(
            [GNNLayer(hidden_dim) for _ in range(gcn_layer_num)]
        )
        self.out = OutLayer(hidden_dim, out_layer_num)
        
    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor):
        """
        Args:
            x: (B, V, 2) nodes_feature (node coords)
            e: (B, E) edges_feature (distance matrix)
            edge_index: (B, 2, E) Tensor with edges representing connections from source to target nodes.
        Returns:
            prob: (B, E, 2) Probability of each edge being connected and not connected to the TSP tour.
        """
        batch_size = x.shape[0]
        e_out = []
        for idx in range(batch_size):
            x_i, e_i = x[idx], e[idx]
            x_i, e_i = self.embed(x_i, e_i)
            for gcn_layer in self.gcn_layers:
                x_i, e_i = gcn_layer(x_i, e_i, edge_index[idx])
            e_i = self.out(e_i)  # shape: (E, 2)
            e_out.append(e_i)
        e_out = torch.stack(e_out, dim=0)  # shape: (B, E, 2)
        return e_out
```

这里，`GCNEncoder` 接收一个 batch 的数据，逐个输入模型，并将模型输出拼接成最后的 Encoder 输出。

{% note warning %}

但注意，这**并不是正确的处理方法**，因为逐个处理会失去并行性的加速。一般做法是，**将一个 batch 中的图拼成一张大图处理**，具体可以参考我解读 GNN4CO 代码的博客。

{% endnote %}

至此，基于 GCN 的 Encoder 网络搭建完毕。完整模型及训练见 [Paradigm 1: Supervised GNN + Decoding](https://cny123222.github.io/2025/07/27/Paradigm-1-Supervised-GNN-Decoding/) 这篇博客。

## 参考资料
[1] C. K. Joshi, T. Laurent, and X. Bresson, “An efficient graph convolutional network technique for the travelling salesman problem,” *arXiv preprint arXiv:1906.01227*, 2019.

[2] 跟李沐学AI,零基础多图详解图神经网络(GNN/GCN),[https://www.bilibili.com/video/BV1iT4y1d7zP](https://www.bilibili.com/video/BV1iT4y1d7zP)

[3] [https://github.com/Thinklab-SJTU/ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)