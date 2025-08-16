---
title: Understading Transformer - An ML4CO perspective
date: 2025-08-01 09:42:27
index_img: img/ml4co.png
tags:
  - AI
  - ML4CO
category:
  - ML4CO
  - Basics
sticky: 300
---

本期讲解 ML4CO 中的常见网络结构 Transformer，并附有代码示例。项目代码可以在 [https://github.com/cny123222/A-Living-Guide-to-ML4CO](https://github.com/cny123222/A-Living-Guide-to-ML4CO) 中找到。

<!-- more -->

## 引言

使用**自回归+强化学习**，也是一类常用的 ML4CO 方法。模型每次输出一个选择，并从最终解的好坏中进行学习——这符合人类的直觉，也**省去了监督学习方法中 decoding 构造最终解的步骤**。

对于自回归模型，一般**选用基于 Attention 的模型**，这里我们采用类似 transformer 的模型进行搭建，主要参考了论文 [1] 中的方法，但进行了一些简化。

我们还是以 TSP 问题为例，整个模型由 Encoder 和 Decoder 两部分组成。

## Encoder 及其搭建

Encoder 由**一个 Embedding 层和 N 个 Attention 层**组成，每个 Attention 层由**一个多头注意力（multi-head attention, MHA）层和一个前馈网络（feed-forward, FF）层**组成。

输入的顶点特征（这里是顶点坐标），先经过 Embedding 层变为 $d_h$ 维的顶点嵌入（node embedding），再经过 N 层注意力层的更新，得到最终的顶点嵌入和图嵌入。我们设输入 Attention 层的向量为 $\mathbf{h}_i^{(0)}$，第 $\ell$ 层输出的顶点嵌入向量为 $\mathbf{h}_i^{(\ell)}$，其中 $\ell \in \{1, \dots, N\}$。

![Encoder 层示意图[1]](encoder.png)

### 多头自注意力层及其搭建

**图中的注意力机制**，与普通注意力机制的不同之处在于，**信息只在相邻节点间传递**。但在 TSP 问题中，我们认为任意两个节点之间都有边，即图是完全图，这时我们采用普通的注意力机制实现即可。这一部分，我们将用 Pytorch **实现多头自注意力机制**。

#### 图注意力机制

这里介绍一下图注意力机制。在图注意力机制中，**每个节点从邻居节点中得到的消息 value 的权重，由该节点的 query 和邻居节点的 key 的匹配度决定**。我们设 key 的维度是 $d_k$，value 的维度是 $d_v$，从而 $\mathbf{k}_i \in \mathbb{R}^{d_k}$，$\mathbf{v}_i \in \mathbb{R}^{d_v}$ 且 $\mathbf{q}_i \in \mathbb{R}^{d_k}$。

给定顶点的嵌入向量 $\mathbf{h}_i$，我们就可以将其投影到对应的 query、key、value 向量：
$$
\mathbf{q}_i = W^Q \mathbf{h}_i, \quad \mathbf{k}_i = W^K \mathbf{h}_i, \quad \mathbf{v}_i = W^V \mathbf{h}_i
$$
其中 $W^Q, W^K \in \mathbb{R}^{d_k \times d_h}$，$W^V \in \mathbb{R}^{d_v \times d_h}$。

接着，我们计算第 $i$ 个顶点的 query 向量 $\mathbf{q}_i$ 和第 $j$ 个顶点的 key 向量 $\mathbf{k}_j$ 的匹配度 $u_{ij} \in \mathbb{R}$，一般用 scaled dot-product 计算：
$$
u_{ij} =
\begin{cases}
\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} \quad \text{if } i \text{ adjcent to } j \\
- \infty \quad \text{otherwise}
\end{cases}
$$
注意在图注意力机制中，我们只在相邻节点间传递信息，不相邻节点间的匹配度设置为 $- \infty$ 以避免信息传递。

然后，我们将刚计算出的注意力分数经过 Softmax 转换为注意力权重（attention weights）$a_{ij} \in [0, 1]$：
$$
a_{ij} = \frac{e^{u_{ij}}}{\sum_{j'}e^{u_{ij'}}}
$$

最终，将各个节点的 value 向量按照注意力权重相加，得到输出：
$$
\mathbf{h}_i' = \sum_{j} a_{ij} \mathbf{v}_j
$$

#### 多头注意力机制

再简单介绍一下多头注意力机制。相比于单头注意力，多头注意力能从输入中捕捉不同方面的信息，有利于特征的提取。

设注意力的头数为 $M$，取 $d_k = d_v = \frac{d_h}{M}$。具体做法是，准备 $M$ 套不同的 $W^Q_m$、$W^K_m$ 和 $W^V_m$ 矩阵，分别按之前的步骤得到输出向量 $\mathbf{h}_{im}'$，拼接后进行一个线性映射，得到最终的输出，即
$$
\mathrm{MHA}_i(\mathbf{h}_1, \dots, \mathbf{h}_n) = \sum_{m=1}^M W_m^O \mathbf{h}_{im}'
$$
其中 $W_m^O \in \mathbb{R}^{d_h \times d_v}$。

#### 代码搭建

我们整体搭建一下多头自注意力层。

```python
# attention/attn_layer.py
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Create separate linear layers for Query, Key, and Value
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        
        # Create the final fully connected output layer
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: Tensor):
        """
        Forward pass for the Multi-Head Self-Attention layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_nodes, embed_dim).
        """
        batch_size = x.shape[0]
        
        # 1. Project input into Q, K, V using separate linear layers
        Q = self.fc_q(x)  # Shape: (batch_size, num_nodes, embed_dim)
        K = self.fc_k(x)  # Shape: (batch_size, num_nodes, embed_dim)
        V = self.fc_v(x)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # 2. Split the embed_dim into num_heads and head_dim
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        
        # 3. Calculate scaled dot-product attention
        # Calculate the dot product of Q and K
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, num_heads, num_nodes, num_nodes)
        # Scale the attention scores
        scaled_attn_scores = attn_scores / math.sqrt(self.head_dim)
        # Apply softmax to get the attention weights
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)  # Shape: (batch_size, num_heads, num_nodes, num_nodes)
        # Multiply the weights by V to get the context vector
        context = torch.matmul(attn_weights, V)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        
        # 4. Concatenate the attention heads' outputs
        # First, transpose to bring num_nodes and num_heads dimensions together
        context = context.transpose(1, 2).contiguous()  # Shape: (batch_size, num_nodes, num_heads, head_dim)
        # Then, reshape to combine the last two dimensions
        context = context.view(batch_size, -1, self.embed_dim)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # 5. Pass the concatenated context vector through the final linear layer
        output = self.fc_out(context)  # Shape: (batch_size, num_nodes, embed_dim)

        return output
```

这段代码中涉及到一些 Tensor 变换操作，如 `.transpose()`、`.view()`，不熟悉的可以参考我的另一篇博客 [Pytorch Tensors: A Beginner's Guide](https://cny123222.github.io/2025/08/16/Pytorch-Tensors-A-Beginner-s-Guide/)。

{% note info %}

注意，这里的代码实现和之前所讲略有不同，我们并没有准备 $M$ 套不同的 $W^Q_m$、$W^K_m$ 和 $W^V_m$ 矩阵，而是通过分别乘一个大的 $W_Q$、$W_K$ 和 $W_V$ 矩阵然后进行拆分来实现的。

{% endnote %}

### 前馈网络层及其搭建

前馈网络由两个线性层组成，中间夹着一个激活函数（如 ReLU）。**与多层感知机（MLP）的核心区别是**，前馈网络对每个元素（即每个节点）进行相同的非线性变换，即**这种变换是节点独立（node-wise）的**。

前馈网络的 Pytorch 实现非常简单，就是两个线性层。

```python
# attention/ff_layer.py
from torch import Tensor, nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super(FeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Create the first linear layer
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # Create the second linear layer
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x: Tensor):
        """
        Forward pass for the Feed Forward Neural Network layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_nodes, embed_dim).
        """
        # Apply the first linear layer followed by ReLU activation
        x = F.relu(self.fc1(x))  # Shape: (batch_size, num_nodes, hidden_dim)
        # Apply the second linear layer
        output = self.fc2(x)  # Shape: (batch_size, num_nodes, embed_dim)
        
        return output
```

### Attention 层及其搭建

有了 `MultiHeadSelfAttention` 和 `FeedForward`，我们可以搭建单层的 Attention 层——只需要加入 **Batch Normalization** 和 **Skip Connection** 这两个技巧。具体来说：
$$
\hat{\mathbf{h}}_i = \mathrm{BN}^\ell\left(\mathbf{h}_i^{(\ell-1)} + \mathrm{MHA}\left(\mathbf{h}_1^{(\ell-1)}, \dots, \mathbf{h}_n^{(\ell-1)}\right)\right)
$$
$$
\mathbf{h}_i^{(\ell)} = \mathrm{BN}^\ell\left(\hat{\mathbf{h}}_i + \mathrm{FF}^\ell(\hat{\mathbf{h}}_i)\right)
$$

以下是对应的代码实现：

```python
# attention/encoder.py
from torch import Tensor, nn
from torch import Tensor, nn
from .attn_layer import MultiHeadSelfAttention
from .ff_layer import FeedForward


class AttentionLayer(nn.Module):
    """
    A single Attention Layer that follows the structure from the image.
    It consists of a Multi-Head Attention sublayer and a Feed-Forward sublayer.
    Each sublayer is followed by a skip connection and Batch Normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        
        # Sublayer 1: Multi-Head Attention
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # Sublayer 2: Feed-Forward Network
        self.ff = FeedForward(embed_dim, hidden_dim)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, x: Tensor):
        """
        Forward pass for one complete attention layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, embed_dim).
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # --- Multi-Head Attention Sublayer ---
        # 1. Apply MHA
        mha_output = self.mha(x)
        
        # 2. Add skip connection
        sublayer1_input = x + mha_output
        
        # 3. Apply Batch Normalization
        # Permute from (batch, seq_len, features) to (batch, features, seq_len) for BN
        sublayer1_output = self.bn1(sublayer1_input.permute(0, 2, 1)).permute(0, 2, 1)

        # --- Feed-Forward Sublayer ---
        # 1. Apply Feed-Forward network
        ff_output = self.ff(sublayer1_output)
        
        # 2. Add skip connection
        sublayer2_input = sublayer1_output + ff_output
        
        # 3. Apply Batch Normalization
        # Permute for BN and then permute back
        output = self.bn2(sublayer2_input.permute(0, 2, 1)).permute(0, 2, 1)

        return output  # Shape: (batch_size, num_nodes, embed_dim)
```

值得注意的是，我们在应用 `BatchNorm1d` 层前，进行了 `.permute(0, 2, 1)` 的操作，即交换了后两个维度，而在 `BatchNorm1d` 层之后，又进行了 `.permute(0, 2, 1)` 的操作，将两个维度交换回来。

这是由于 `nn.BatchNorm1d` **默认第二个维度是特征维度**（num_features，即这里的 `embed_dim`），并且沿这个维度计算均值和方差，以进行归一化。而我们的实现中 `embed_dim`	 在最后一个维度，因此需要交换到第二个维度以满足 `nn.BatchNorm1d` 对输入的要求。

### Encoder 整体搭建

有了以上的工作，Encoder 的整体搭建变得十分简单，只要用一个 Embedding 层将顶点坐标映射到 `embed_dim` 维，再通过 N 个 Attention 层即可。代码实现如下：

```python
# attention/encoder.py
class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int):
        super(AttentionEncoder, self).__init__()
        # Embedding layer
        self.embed = nn.Linear(2, embed_dim)
        
        # Stack of identical Attention Layers
        self.layers = nn.ModuleList([
            AttentionLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x: Tensor):
        """
        Forward pass for the Encoder.
        Args:
            x (torch.Tensor): Coordinates of nodes with shape (batch_size, num_nodes, 2).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_nodes, embed_dim).
        """
        # Embed the input coordinates
        x = self.embed(x)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # Pass through multiple attention layers
        for layer in self.layers:
            x = layer(x)  # Shape: (batch_size, num_nodes, embed_dim)
            
        return x  # Shape: (batch_size, num_nodes, embed_dim)
```

## Decoder 及其搭建

接着我们进入 Decoder 的搭建。Decoder 负责**在每个时间步 $t \in \{1, \dots, n\}$ 下，根据 Encoder 和已选顶点 $\pi_{t'}(t' < t)$ 的信息，生成下一个选择的顶点 $\pi_t$**。

具体来说，Decoder 要做的事是：
- **Step 1:** 根据当前状况生成一个 context embedding 向量 $\mathbf{h}_{(c)}^{(N)}$；
- **Step 2:** 以 $\mathbf{h}_{(c)}^{(N)}$ 作为 query，和所有还未选择节点的 $\mathbf{h}_i^{(N)}$ 作为 key 进行匹配，通过多头交叉注意力 glimpse 一下所有候选节点情况，更新到 $\mathbf{h}_{(c)}^{(N+1)}$；
- **Step 3:** 使用更新后的 $\mathbf{h}_{(c)}^{(N+1)}$ 作为 query，使用单头注意力计算各个未选择节点的注意力分数，生成下一步选择各个顶点的概率值。

![Decoder 层示意图[1]](decoder.png)

### 带掩码的多头交叉注意力层及其搭建

{% note info %}

这里将实现 Decoder 任务的 **Step 2**。

{% endnote %}

与 Encoder 中的自注意力不同，Decoder 中采用**交叉注意力机制**，用一个全局的 context embedding 向量 $\mathbf{h}_{(c)}^{(N)} \in \mathbb{R}^{d_h}$ 作为 query（其构造方法我们稍后介绍），以 Encoder 的输出作为 key 和 value，并**用一个 mask 屏蔽了已经选过的节点**，得到更新的 $\mathbf{h}_{(c)}^{(N+1)}$。

具体来说，先通过线性映射得到 query、key、value 向量：
$$
\mathbf{q}_{(c)} = W^Q \mathbf{h}_{(c)}, \quad \mathbf{k}_i = W^K \mathbf{h}_i, \quad \mathbf{v}_i = W^V \mathbf{h}_i
$$

接着计算注意力分数：
$$
u_{(c)j} = 
\begin{cases}
\frac{\mathbf{q}_{(c)}^T \mathbf{k}_j}{\sqrt{d_k}} \quad \text{if } j \neq \pi_{t'}, \forall t' < t \\
- \infty \quad \text{otherwise}
\end{cases}
$$

注意到，我们将已经访问过节点的分数设置为了 $-\infty$，这样在 Softmax 后其注意力权重就会变为 $0$。

接着，我们搭建这个带掩码的多头交叉注意力层。

```python
# attention/decoder.py
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiHeadMaskedCrossAttention(nn.Module):
    """
    Implements a Multi-Head Cross-Attention layer with masking.

    This layer is designed for a decoder that needs to attend to the output of an
    encoder. It takes a single context vector as the query source and a sequence
    of encoder outputs as the key and value source. It also supports masking to
    prevent attention to nodes that have already been visited in TSP.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadMaskedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Query, Key, Value, and the final output projection
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, context_query: Tensor, encoder_outputs: Tensor, mask: Tensor = None):
        """
        Forward pass for the Multi-Head Masked Cross-Attention layer.

        Args:
            context_query (torch.Tensor): The query tensor, typically derived from the decoder's state.
                                          Shape: (batch_size, 1, embed_dim).
            encoder_outputs (torch.Tensor): The key and value tensor, typically the output from the encoder.
                                            Shape: (batch_size, num_nodes, embed_dim).
            mask (torch.Tensor, optional): A boolean or 0/1 tensor to mask out certain keys.
                                           A value of 0 indicates the position should be masked.
                                           Shape: (batch_size, num_nodes).

        Returns:
            output (torch.Tensor): The attention-weighted output vector.
                                    Shape: (batch_size, 1, embed_dim).
        """
        batch_size = context_query.shape[0]
        num_nodes = encoder_outputs.shape[1]

        # 1. Project Q from the context query and K, V from the encoder outputs.
        Q = self.fc_q(context_query)    # Shape: (batch_size, 1, embed_dim)
        K = self.fc_k(encoder_outputs)  # Shape: (batch_size, num_nodes, embed_dim)
        V = self.fc_v(encoder_outputs)  # Shape: (batch_size, num_nodes, embed_dim)

        # 2. Reshape and transpose for multi-head processing.
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)          # Shape: (batch_size, num_heads, 1, head_dim)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # Shape: (batch_size, num_heads, num_nodes, head_dim)

        # 3. Compute scaled dot-product attention scores.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, num_heads, 1, num_nodes)

        # 4. Apply the mask before the softmax step.
        if mask is not None:
            # Reshape mask for broadcasting: (batch_size, num_nodes) -> (batch_size, 1, 1, num_nodes)
            mask_reshaped = mask.unsqueeze(1).unsqueeze(2)
            # Fill masked positions (where mask is 0) with a very small number.
            attn_scores = attn_scores.masked_fill(mask_reshaped == 0, -1e9)
        
        # 5. Scale scores, apply softmax, and compute the context vector.
        scaled_attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scaled_attn_scores, dim=-1) # Shape: (batch_size, num_heads, 1, num_nodes)
        context = torch.matmul(attn_weights, V)              # Shape: (batch_size, num_heads, 1, head_dim)

        # 6. Concatenate heads and pass through the final linear layer.
        context = context.transpose(1, 2).contiguous()  # Shape: (batch_size, 1, num_heads, head_dim)
        context = context.view(batch_size, 1, self.embed_dim)  # Shape: (batch_size, 1, embed_dim)
        output = self.fc_out(context)  # Shape: (batch_size, 1, embed_dim)

        return output
```

### Decoder 整体搭建

我们离 Decoder 的搭建完成还剩两个部分：Context embedding 的构建、最终节点概率的计算。

#### Context Embedding 的构建

{% note info %}

这里将实现 Decoder 任务的 **Step 1**。

{% endnote %}

Context Embedding 作为每个时间步 $t$ 中选择节点的 query，其构建方法至关重要。模型应当知道**图的全局信息、起始节点信息、前一节点信息**等，以便做出下一个节点的选择。在 AM 模型[1]中，作者采用的构建方式是：
$$
\mathbf{h}_{(c)}^{(N)} = 
\begin{cases}
[\bar{\mathbf{h}}^{(N)}, \mathbf{h}_{\pi_{t-1}}^{(N)}, \mathbf{h}_{\pi_1}^{(N)}] &\quad t > 1 \\
[\bar{\mathbf{h}}^{(N)}, \mathbf{v}^1, \mathbf{v}^\mathrm{f}] &\quad t = 1
\end{cases}
$$
其中 $[\cdot,\cdot,\cdot]$ 表示  concatenation，即将三个 $d_h$ 维的向量拼接成一个 $3 \cdot d_h$ 维的向量。

$\bar{\mathbf{h}}^{(N)}$ 是 graph embedding 向量，表征了**图的全局信息**，其计算方式是最终所有 node embedding 向量的平均值，即：
$$
\bar{\mathbf{h}}^{(N)} = \frac{1}{n} \sum_{i=1}^n \mathbf{h}_i^{(N)}
$$

在第一次选择时，我们将 graph embedding 和两个可学习的 $d_h$ 维 placeholder 向量 $\mathbf{v}^1$、$\mathbf{v}^\mathrm{f}$ 拼接，得到 context embedding $\mathbf{h}_{(c)}^{(N)}$。而在后续选择时，我们将 graph embedding、前一个选择节点的 node embedding、起始节点的 node embedding 拼接，得到完整的 context embedding 向量。

{% note secondary %}

与原文稍有不同的是，我们的实现中在得到 $3\cdot d_h$ 维的向量 $\mathbf{h}_{(c)}^{(N)}$ 之后，先将其映射到 $d_h$ 维，再输入多头交叉注意力层。原文[1]中的处理是直接输入注意力层，通过 $W^Q$ 形状的改变实现。两种方法效果是一样的。

{% endnote %}

#### 对数概率值计算

{% note info %}

这里将实现 Decoder 任务的 **Step 3**。

{% endnote %}

多头交叉注意力层使得 context embedding 向量 $\mathbf{h}_{(c)}^{(N+1)}$ 包含了 glimpse 各节点向量得到的信息。在得到了 $\mathbf{h}_{(c)}^{(N+1)}$ 之后，我们再使用一个单头注意力，计算每个候选节点的概率。

具体来说，是以 $\mathbf{h}_{(c)}^{(N+1)}$ 为 query，各节点的 node embedding 为 key，计算注意力分数。不同之处是，我们对注意力分数做了剪裁，剪裁到 $[-C, C]$，得到对数概率值（logits），具体来说：
$$
u_{(c)j} = 
\begin{cases}
C \cdot \tanh\left(\dfrac{\mathbf{q}_{(c)}^T \mathbf{k}_j}{\sqrt{d_k}}\right) &\quad \text{if } j \neq \pi_{t'}, \forall t' < t \\
- \infty &\quad \text{otherwise}
\end{cases}
$$

最终使用 Softmax 计算每个节点被选择的概率：
$$
p_i = p_\mathbf{\theta}(\pi_t = i \vert s, \pi_{1:t-1}) = \frac{e^{u_{(c)i}}}{\sum_j e^{u_{(c)j}}}
$$

在这里，我们只返回对数概率值，Softmax 的计算交给外层函数处理。

#### 代码构建

```python
# attention/decoder.py
class AttentionDecoder(nn.Module):
    """
    Implements the Decoder for the Attention Model.

    At each step, it creates a context embedding based on the graph, the first
    node, and the previously visited node. It then uses two attention mechanisms:
    1. A multi-head "glimpse" to refine the context.
    2. A single-head mechanism with clipping to calculate the final output probabilities.
    """
    def __init__(self, embed_dim: int, num_heads: int, clip_value: float = 10.0):
        super(AttentionDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.clip_value = clip_value

        # Learned placeholders for the first and last nodes at the initial step (t=1)
        self.v_first_placeholder = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.v_last_placeholder = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Projection layer for the concatenated context vector
        self.context_projection = nn.Linear(3 * embed_dim, embed_dim, bias=False)

        # The first attention mechanism: a multi-head "glimpse".
        self.glimpse_attention = MultiHeadMaskedCrossAttention(embed_dim, num_heads)
        
        # Layers for the final single-head attention mechanism to compute probabilities.
        self.final_q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.final_k_projection = nn.Linear(embed_dim, embed_dim, bias=False)


    def forward(self, encoder_outputs: Tensor, partial_tour: Tensor, mask: Tensor):
        """
        Performs a single decoding step.

        Args:
            encoder_outputs (torch.Tensor): The final node embeddings from the encoder.
                                            Shape: (batch_size, num_nodes, embed_dim).
            partial_tour (torch.Tensor): A tensor of node indices for the current partial tours.
                                         Shape: (batch_size, current_tour_length).
            mask (torch.Tensor): A tensor indicating which nodes are available to be visited.
                                 Shape: (batch_size, num_nodes).

        Returns:
            log_probs (torch.Tensor): The log-probabilities for selecting each node as the next step.
                                        Shape: (batch_size, num_nodes).
        """
        batch_size = encoder_outputs.shape[0]

        # Step 1: Construct the Context Embedding for the entire batch
        graph_embedding = encoder_outputs.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1, embed_dim)

        if partial_tour.size(1) == 0: # If this is the first step (t=1) for all instances
            # Use learned placeholders
            first_node_emb = self.v_first_placeholder.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
            last_node_emb = self.v_last_placeholder.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        else:
            # Get indices of the first and last nodes for each instance in the batch
            first_node_indices = partial_tour[:, 0]  # Shape: (batch_size,)
            last_node_indices = partial_tour[:, -1]  # Shape: (batch_size,)
            
            # Use torch.gather to select the corresponding embeddings
            first_node_emb = torch.gather(encoder_outputs, 1, first_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim))  # Shape: (batch_size, 1, embed_dim)
            last_node_emb = torch.gather(encoder_outputs, 1, last_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim))  # Shape: (batch_size, 1, embed_dim)

        # Concatenate the three components to form the raw context
        raw_context = torch.cat([graph_embedding, first_node_emb, last_node_emb], dim=2)  # Shape: (batch_size, 1, 3 * embed_dim)
        
        # Project the context to create the initial query
        context_query = self.context_projection(raw_context)  # Shape: (batch_size, 1, embed_dim)

        # Step 2: Perform the Multi-Head "Glimpse"
        glimpse_output = self.glimpse_attention(
            context_query=context_query,
            encoder_outputs=encoder_outputs,
            mask=mask
        )  # Shape: (batch_size, 1, embed_dim)

        # Step 3: Calculate Final Log-Probabilities
        final_q = self.final_q_projection(glimpse_output)  # Shape: (batch_size, 1, embed_dim)
        final_k = self.final_k_projection(encoder_outputs)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # Calculate compatibility scores (logits)
        logits = torch.matmul(final_q, final_k.transpose(-2, -1)).squeeze(1) / math.sqrt(self.embed_dim)  # Shape: (batch_size, num_nodes)

        # Clip the logits before masking
        logits = self.clip_value * torch.tanh(logits)  # Shape: (batch_size, num_nodes)

        # Apply the mask again to ensure forbidden nodes are not chosen
        logits[mask == 0] = -torch.inf

        # Return the log-probabilities
        return logits  # Shape: (batch_size, num_nodes)
```

这段代码完全按照我们刚才所说逻辑。其中比较难理解的是：
```python
first_node_emb = torch.gather(encoder_outputs, 1, first_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim))  # Shape: (batch_size, 1, embed_dim)
last_node_emb = torch.gather(encoder_outputs, 1, last_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim))  # Shape: (batch_size, 1, embed_dim)
```

我们以 `first_node_emb` 的计算为例，详细讲一下 `torch.gather()` 操作。

首先，`first_node_indices` 是一个形状为 `(batch_size,)` 的 Tensor，存放了这个 batch 中每个 instance 路径的起始节点序号。`encoder_outputs` 是 Encoder 的输出，形状为 `(batch_size, num_nodes, embed_dim)`，其中有每个节点的 embedding 向量。我们要做的事是，把每个起始节点序号的 embedding 向量从 `encoder_outputs` 中取出，组成一个形状为 `(batch_size, 1, embed_dim)` 的 Tensor。

我们来看 `torch.gather()` 的工作原理，其接收了三个参数 `input=encoder_outputs`、`dim=1` 和 `index=first_node_indices.view(-1, 1, 1).expand(-1, -1, self.embed_dim)`。`torch.gather()` 会拿着 `index` 中的每个索引，在 `dim` 维度上对 `input` 取数，最终得到和 `index` 形状完全相同的 Tensor。

{% fold info @举一个简单的例子： %}
```python
input = torch.tensor([[10, 20, 30, 40],     # Row 0
                      [50, 60, 70, 80],     # Row 1
                      [90, 100, 110, 120]]) # Row 2
index = torch.tensor([[0, 2],    # For Row 0, get items from column 0 and column 2
                      [3, 1],    # For Row 1, get items from column 3 and column 1
                      [1, 3]])   # For Row 2, get items from column 1 and column 3
output = torch.gather(input, dim=1, index=index)
```
首先，`gather` 创建一个和 `index` 形状完全相同的 `output` 张量，即 `output` 的形状也会是 `(3, 2)`。

接着，`gather` 函数会按照 `index` 的指引，逐一去 `input` 中取数。`index[i][j]` 的值指明了 `dim` 维的索引，其余维度按照 `index` 的位置进行。如对于 `dim=1`，`output[i][j] = input[i][index[i][j]]`。

这个例子中：
- `index[0][0] = 0`，表明 `output[0][0]` 要从第 0 行的第 0 个位置取数，从而 `output[0][0] = 10`。
- `index[0][1] = 2`，表明 `output[0][1]` 要从第 0 行的第 2 个位置取数，从而 `output[0][0] = 30`。
- `index[1][0] = 3`，表明 `output[1][0]` 要从第 1 行的第 3 个位置取数，从而 `output[1][0] = 80`。
- `index[1][1] = 1`，表明 `output[1][1]` 要从第 1 行的第 1 个位置取数，从而 `output[1][1] = 60`。
- `index[2][0] = 1`，表明 `output[2][0]` 要从第 2 行的第 1 个位置取数，从而 `output[2][0] = 100`。
- `index[2][1] = 3`，表明 `output[2][1]` 要从第 2 行的第 3 个位置取数，从而 `output[2][1] = 120`。

可以验证结果：
```python
print(output)
# tensor([[ 10,  30],
#         [ 80,  60],
#         [100, 120]])
```

{% endfold %}

在这里，维度数为 3 且参数 `dim=1`，因此有 `output[i][j][k] = input[i][index[i][j][k]][k]`。

由于 `torch.gather()` 要求提供给它的 `index` 的维度数必须和 `input` 的维度数相同，我们先对 `first_node_indices` 进行升维。使用 `.view(-1, 1, 1)`，我们将其维度升高到 3 维，与 `input` 维度数匹配。这时的形状为 `(batch_size, 1, 1)`。

接着，我们采用 `.expand(-1, -1, self.embed_dim)`，目的是告诉 `gather`，对于我们想要抽取的那个节点，我们想要它的全部 `embed_dim` 个特征。`expand` 函数沿着大小为 1 的维度进行复制，将其形状变为 `(batch_size, 1, embed_dim)`。

举个例子，设 `first_node_indices = tensor([0, 1, 2])`，`embed_dim = 4`。那么经过 `.view(-1, 1, 1)` 后，变为 `tensor([[[0]], [[1]], [[2]]])`，再经过 `.expand(-1, -1, self.embed_dim)` 后，变为 `tensor([[[0, 0, 0, 0]], [[1, 1, 1, 1]], [[2, 2, 2, 2]]])`。将这个 Tensor 作为 `index`	，就是告诉 `gather()`，对于第 0 个 instance（instance 在第 0 个维度），每一维特征（特征在第 2 个维度）都在节点维度（节点在第 1 个维度）上取第 0 个节点的数值；对于第 1 个 instance，每一维特征都在节点维度上取第 1 个节点的数值；对于第 2 个 instance，每一维特征都在节点维度上取第 2 个节点的数值。最终就得到了每个 instance 的起始节点的全部维度的特征向量。

至此，我们完成了基于 Attention 的 Encoder 和 Decoder 的搭建。完整模型及训练见 [Paradigm 2: Autoregressive Transformer + RL](https://cny123222.github.io/2025/08/01/Paradigm-2-Autoregressive-Transformer-RL/) 这篇博客。

## 参考资料
[1] W. Kool, H. Van Hoof, and M. Welling, “Attention, learn to solve routing problems!” *arXiv preprint arXiv*:1803.08475, 2018.
[2] [https://github.com/Thinklab-SJTU/ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)