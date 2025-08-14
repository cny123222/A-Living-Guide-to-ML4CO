---
title: 'Paradigm 2: Autoregressive Transformer + RL'
date: 2025-08-01 09:42:56
index_img: img/ml4co.png
tags:
  - AI
  - ML4CO
category:
  - ML4CO
  - Basics
sticky: 300
---

本文将介绍 ML4CO 的第二种范式（Transformer 自回归 + 强化学习）的代码实现。代码文件可以在 [https://github.com/cny123222/A-Living-Guide-to-ML4CO](https://github.com/cny123222/A-Living-Guide-to-ML4CO) 中找到。

<!-- more -->

## 方法概述

在 [A Living Guide to ML4CO](https://cny123222.github.io/2025/07/25/A-Living-Guide-to-ML4CO) 中，我们提到过 ML4CO 的第二种常见范式：**Transformer 自回归 + 强化学习**。它一步步构建问题的解，并从解的好坏中进行学习。

我们还是以 TSP 为例，这种方法的大致过程是：

- **数据准备**：强化学习**不需要准备已知的最优解**
- **模型构建**：策略网络一个节点一个节点地构建出最终的路径，在第 `t` 步时，
  - **输入**：Transformer 会接收**当前已经构建的部分路径以及所有节点的全局信息**作为输入
  - **输出**：模型输出是在所有尚未访问的节点中，**选择下一个要访问的节点的概率分布**
- **训练过程**：Transformer 构建出一条完整路径后，计算总长度，用**总长度作为奖励信号**，使用 REINFORCE 进行更新

其中，“**模型构建**”部分由于涉及 Attention 和 Transformer 的知识，我们将其放在了 [Understading Transformer - An ML4CO perspective](https://cny123222.github.io/2025/08/01/Understading-Transformer-An-ML4CO-perspective/) 这篇博客中。我们用 Pytorch 搭建了基于 Attention 的 Encoder 和 Decoder 网络，网络结构参考论文[1]。

这篇博客中，我们将直接使用上述 Encoder 和 Decoder 网络，实现这种方法完整的训练过程。

## 数据准备

我们沿用 [Paradigm 1: Supervised GNN + Decoding](https://cny123222.github.io/2025/07/27/Paradigm-1-Supervised-GNN-Decoding/) 中生成的数据集。其训练集、验证集和测试集位于 `data/tsp20` 路径下，格式为 `.txt`，且均有 LKH 求解器生成的最优解。事实上，RL 的训练过程**不需要用到参考的最优解**。

`TSPDataset` 部分沿用 [Paradigm 1: Supervised GNN + Decoding](https://cny123222.github.io/2025/07/27/Paradigm-1-Supervised-GNN-Decoding/) 中的 `TSPDataset` 代码。不同之处在于，这里 RL 中的 Dataset 只返回节点坐标 `points` 和参考解 `ref_tours`（为了计算 gap）。

```python
# attention/env.py
import numpy as np
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    def __init__(self, file_path: str):
        # read the data form .txt
        with open(file_path, "r") as file:
            points_list = list()
            tour_list = list()
            for line in file:
                line = line.strip()
                split_line = line.split(" output ")
                # parse points
                points = split_line[0].split(" ")
                points = np.array([[float(points[i]), float(points[i + 1])] 
                                   for i in range(0, len(points), 2)])
                points_list.append(points)
                # parse tour
                tour = split_line[1].split(" ")
                tour = np.array([int(t) for t in tour])
                tour -= 1  # convert to 0-based index
                tour_list.append(tour)
        self.points = np.array(points_list)
        self.tours = np.array(tour_list)
        
        # Convert to tensors
        self.points = torch.tensor(self.points, dtype=torch.float32)  # Shape: (num_samples, num_nodes, 2)
        self.tours = torch.tensor(self.tours, dtype=torch.long)  # Shape: (num_samples, num_nodes + 1)

    def __getitem__(self, index):
        return self.points[index], self.tours[index]  # Shape: (V, 2) and (V+1,)
    
    def __len__(self):
        return self.points.shape[0]  # number of samples
```

注意，这里的 `tours` 都包含了最终回到起点。

## Environment 搭建

这里我们搭建一个 Env，完成两个功能：**DataLoader 的功能，以及 RL 中 environment 的功能**。回顾一下 RL 中 environment 的功能：接收 agent 的 action，根据 action 改变自身的 state，并将新的 state 和 reward 返回给 agent。我们定义的损失函数：
$$
\mathcal{L}(\mathbf{\theta}\vert s) = \mathbb{E}_{p_\mathbf{\theta}(\mathbf{\pi} \vert s)}[L(\mathbf{\pi})]
$$
即回路 $\mathbf{\pi}$ 的长度的期望。因此，**返回的 reward 就是回路总长度的相反数**。

我们先看代码实现：

```python
# attention/env.py
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ml4co_kit import BaseEnv


@dataclass
class StepState:
    """
    A data class to hold the state of the environment at each decoding step.
    This makes passing state information to the model cleaner.
    """
    current_node: Tensor = None  # Shape: (batch,)
    tours: Tensor = None  # Shape: (batch, time_step)
    mask: Tensor = None  # Shape: (batch, num_nodes)
    

class AttentionEnv(BaseEnv):
    def __init__(
        self,
        mode: str = "train",
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        train_path: str = None,
        val_path: str = None,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        super(AttentionEnv, self).__init__(
            name="AttentionEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_path=train_path,
            val_path=val_path,
            num_workers=num_workers,
            device=device
        )
        if mode is not None:
            self.load_data()
        self.num_nodes = self.train_dataset.points.shape[1] if self.train_dataset else None
        self.points = None
        self.batch_size = None
        # These will be managed during reset and step
        self.current_node = None
        self.tours = None
        self.mask = None

    def load_data(self):
        self.train_dataset = TSPDataset(self.train_path) if self.train_path else None
        self.val_dataset = TSPDataset(self.val_path) if self.val_path else None
        
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def reset(self, points: Tensor):
        """
        Resets the environment for a new rollout.
        """
        self.points = points.to(self.device)  # Shape: (batch_size, num_nodes, 2)
        self.batch_size = self.points.size(0)
        self.current_node = None
        self.tours = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)
        self.mask = torch.ones((self.batch_size, self.num_nodes), device=self.device)
        state_step = StepState(current_node=self.current_node, tours=self.tours, mask=self.mask)
        return state_step, None, None  # Initial state, no reward, not done

    def step(self, selected_node: Tensor):
        """
        Updates the environment state based on the selected node.
        Args:
            selected_node (Tensor): The node selected by the policy model.
                                    Shape: (batch_size,).
        Returns:
            A tuple containing:
            - state (StepState): The new state of the environment.
            - reward (Tensor or None): The final reward (negative tour length) if done, else None.
            - done (bool): A boolean indicating if the tour is complete.
        """
        self.current_node = selected_node
        self.tours = torch.cat([self.tours, self.current_node.unsqueeze(-1)], dim=1)
        self.mask.scatter_(dim=1, index=self.current_node.unsqueeze(-1), value=0)  # Mark the selected node as visited
        
        done = (self.tours.size(1) == self.num_nodes)
        reward = -self.evaluate() if done else None  # Negative tour length as reward
        state_step = StepState(current_node=self.current_node, tours=self.tours, mask=self.mask)
        return state_step, reward, done
        
    def evaluate(self):
        """
        Calculates the total length of the generated tours.

        Returns:
            Tensor: The total length for each tour in the batch. Shape: (batch_size,).
        """
        # Gather coordinates in tour order.
        # self.tours.shape: (batch_size, num_nodes)
        tour_coords = torch.gather(input=self.points, dim=1, index=self.tours.unsqueeze(-1).expand(-1, -1, 2))  # Shape: (batch_size, num_nodes, 2)
        
        # Calculate distances between consecutive nodes, including returning to the start
        rolled_coords = tour_coords.roll(dims=1, shifts=-1)
        segment_lengths = torch.norm(tour_coords - rolled_coords, dim=2)
        
        return segment_lengths.sum(dim=1)
```

`DataLoader` 部分的函数与之前一致，在此从略。

`StepState` 表示 environment 的 **state**，包装成了 data class，相当于 C++ 中的**结构体**。其中包含 `current_node` 当前选中的节点、`tours` 当前构建的部分路径、`mask` 当前掩码（已经访问过的节点为 0，其余节点为 1）。

`reset()` 方法在 rollout 开始前对 state 进行重置。`step()` 方法接收 agent 节点选择的 action，并更新 state，返回 reward。`done` 为 `True` 表示一次路径构建完成，此时返回 `reward` 为回路长度的相反数。`evaluate()` 方法用于计算回路的长度。

{% note info %}

这里代码还是涉及到一些 Tensor 的操作，可以参考另一篇博客 [Fancy but Useful Tensor Operations](https://cny123222.github.io/2025/07/30/Fancy-but-Useful-Tensor-Operations/)。

{% endnote %}

## Policy 网络搭建

Policy 网络的作用是作为 agent，和 environment 进行交互，利用得到的结果更新网络参数。具体来说，Policy 网络从 environment 获取当前的 state，根据 decoder 网络作出下一个节点的选择并告诉 environment，再从 environment 获得 reward 和新的 state。注意，TSP 的 reward 只有在 tour 构建完成的时候，即一轮 rollout 结束的时候，才会获得。

Policy 网络会返回 reward 和 tour 给上层的 Model，Model 会负责训练和验证的具体过程。此外，还需要返回一个 `sum_log_probs`，即各次选择概率的对数之和，将用于 Policy Gradient 方法，这会在下一部分中介绍。

```python
# attention/policy.py
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from .env import AttentionEnv
from .encoder import AttentionEncoder
from .decoder import AttentionDecoder


@dataclass
class StepState:
    """
    A data class to hold the state of the environment at each decoding step.
    This makes passing state information to the model cleaner.
    """
    current_node: Tensor = None  # Shape: (batch,)
    tours: Tensor = None  # Shape: (batch, time_step)
    mask: Tensor = None  # Shape: (batch, num_nodes)
    

class AttentionPolicy(nn.Module):
    def __init__(
        self,
        env: AttentionEnv,
        encoder: AttentionEncoder,
        decoder: AttentionDecoder,
    ):
        super(AttentionPolicy, self).__init__()
        self.env = env
        self.encoder = encoder
        self.decoder = decoder
        self.to(self.env.device)
        
    def forward(self, points: Tensor, mode: str = "sampling"):
        """
        Performs a full rollout to generate a tour for a batch of TSP instances.

        Args:
            points (torch.Tensor): Node coordinates for the batch.
                                    Shape: (batch_size, num_nodes, 2).
            mode (str): 'sampling' for stochastic rollout or 'greedy' for deterministic.

        Returns:
            A tuple containing:
            - reward (torch.Tensor): Reward for each instance in the batch. Shape: (batch_size,).
            - sum_log_probs (torch.Tensor): Sum of action log probabilities. Shape: (batch_size,).
            - tour (torch.Tensor): The decoded tour for each instance. Shape: (batch_size, num_nodes + 1).
        """
        batch_size = points.size(0)
        
        # Pre-computation step
        encoder_outputs = self.encoder(points)  # Shape: (batch_size, num_nodes, embed_dim)
        
        # Initialize environment for this rollout
        state, reward, done = self.env.reset(points)
        
        # Perform the rollout
        sum_log_probs = torch.zeros(batch_size, device=self.env.device)
        while not done:
            log_probs = self.decoder(encoder_outputs, state.tours, state.mask)  # Shape: (batch_size, num_nodes)
            dist = Categorical(logits=log_probs)  # Create a categorical distribution from log probabilities
            if mode == "sampling":
                # Sample from the distribution
                selected_node = dist.sample()  # Shape: (batch_size,)
            elif mode == "greedy":
                selected_node = log_probs.argmax(dim=1)
            else:
                raise NotImplementedError(f"Mode '{mode}' is not implemented.")
            
            sum_log_probs += dist.log_prob(selected_node)
            state, reward, done = self.env.step(selected_node)
            
        tour = state.tours  # Shape: (batch_size, num_nodes)
        start_node = tour[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)
        tour = torch.cat([tour, start_node], dim=1)  # Append the start node to the end of the tour
            
        return reward, sum_log_probs, tour
```

这里写了**两种选择节点的方法**，分别是 `sampling` 和 `greedy`。`sampling` 根据概率分布采样，用于训练时的探索；`greedy` 选择概率最大的节点，用于验证及 baseline 计算（后续介绍）。

## Model 搭建

我们使用强化学习中的 **REINFORCE 算法**进行策略网络的更新。这里，我们详细推导一下论文中 Equation(9)，即 TSP 中的策略梯度公式，顺便复习一下 RL 的相关知识。

### 0. 符号定义

首先我们定义相关符号。TSP 问题实例 $s$ 是一个有 $n$ 个节点的图，节点编号 $i \in \{1, \dots, n\}$。一个解（即 tour）$\mathbf{\pi} = (\pi_1, \dots, \pi_n)$ 是节点的一个全排列，即满足 $\pi_t \in \{1, \dots, n\}$ 且 $\pi_t \neq \pi_{t'}, \forall t \neq t'$。我们的 encoder-decoder 网络，参数为 $\mathbf{\theta}$，选出某条特定路径 $\mathbf{\pi}$ 的概率 $p(\mathbf{\pi}\vert s)$ 可以写为：
$$
p_{\mathbf{\theta}}(\mathbf{\pi}\vert s) = \prod_{t=1}^n p_{\mathbf{\theta}}(\pi_t\vert s, \mathbf{\pi}_{1:t-1})
$$

### 1. 目标 - 我们要优化什么？

我们的目标是找到一个最优的策略网络，即找到最优的参数 $\mathbf{\theta}$，使得这个网络在接收一个问题实例 $s$ 后，能给出一个最优路径 $\mathbf{\pi}$。我们用一个成本函数 $L(\mathbf{\pi})$ 衡量解的好坏，对于 TSP 问题，它就是路径总长度。我们的目标就是要最小化这个成本。当然，因为策略是随机的，我们不能只优化某一条路径 $\mathbf{\pi}$ 的成本，而应该**优化所有可能路径的期望成本**。因此，**我们的目标函数**是：
$$
\mathcal{L}(\mathbf{\theta}\vert s) = \mathbb{E}_{\mathbf{\pi} \sim p_{\mathbf{\theta}}(\mathbf{\pi}\vert s)}[L(\mathbf{\pi})] = \sum_{\mathbf{\pi}} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)L(\mathbf{\pi})
$$
这个公式的含义是：把所有可能的路径 $\mathbf{\pi}$ 的成本 $L(\mathbf{\pi})$，用它被策略网络选中的概率 $p_{\mathbf{\theta}}(\mathbf{\pi}\vert s)$ 进行加权平均。总的来说，我们的目标就是通过调整参数 $\mathbf{\theta}$ 来最小化这个期望成本。

### 2. 挑战 - 如何对期望求梯度？

为了用梯度下降法最小化 $\mathcal{L}(\mathbf{\theta}\vert s)$，我们需要计算它对参数 $\mathbf{\theta}$ 的梯度 $\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}\vert s)$。直接求梯度，会得到：
$$
\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}\vert s) = \nabla_{\mathbf{\theta}} \sum_{\mathbf{\pi}} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)L(\mathbf{\pi}) = \sum_{\mathbf{\pi}} \nabla_{\mathbf{\theta}}p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)L(\mathbf{\pi})
$$
这个梯度的计算**非常困难**，具体来说，会有两方面的挑战：

首先，**梯度项 $\nabla_{\mathbf{\theta}}p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$ 的形式非常复杂**。$p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$ 的含义是给定问题实例 $s$，策略网络选择路径 $\mathbf{\pi}$ 的概率，其计算方法是：
$$
p_{\mathbf{\theta}}(\mathbf{\pi}|s) = p_{\mathbf{\theta}}(\pi_1|s)p_{\mathbf{\theta}}(\pi_2|s, \pi_1)p_{\mathbf{\theta}}(\pi_3|s, \pi_1, \pi_2) \cdots p_{\mathbf{\theta}}(\pi_n|s, \mathbf{\pi}_{1:n-1})
$$
即 $p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$ 是**一长串概率的连乘积**，其中每一项是第 $t$ 步时模型 Softmax 层输出的概率。

根据**乘积的求导法则**，对这个式子求导将会得到一个极其冗长和复杂的表达式，非常不方便计算。

其次，**这个梯度包含一个无法用采样近似的求和**。根据这个梯度的表达式，我们需要对所有可能的路径 $\mathbf{\pi}$ 求和。这显然是不现实的，因为可能的路径数量过于庞大。因此，我们**一般用采样的方法近似**。不幸的是，这个表达式无法直接采样近似。原因是：我们采样依据的分布是 $p_{\mathbf{\theta}}$，只能得到按 $p_{\mathbf{\theta}}$ 加权平均的结果，而现在的公式 $\sum_{\mathbf{\pi}} (\nabla_{\mathbf{\theta}}p_{\mathbf{\theta}})\cdot L(\mathbf{\pi})$ 是按梯度 $\nabla_{\mathbf{\theta}}p_{\mathbf{\theta}}$ 的加权平均，我们无法通过采样进行估计。

总的来说，直接计算梯度 $\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}\vert s)$ 非常棘手。

### 3. 技巧 - Log-Derivative Trick

为了解决这个问题，我们引入一个很巧妙的 trick。首先回顾一下对数的求导方法：
$$
\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}
$$
移项可得：
$$
\nabla_x f(x) = f(x) \nabla_x \log f(x)
$$
令 $f(x) = p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$：
$$
\nabla_\mathbf{\theta} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) = p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) \nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)
$$
代入我们之前的梯度公式：
$$
\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}\vert s) = \sum_{\mathbf{\pi}} \nabla_{\mathbf{\theta}}p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)L(\mathbf{\pi}) = \sum_{\mathbf{\pi}} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) \nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)L(\mathbf{\pi})
$$
把求和符号 $\sum_\mathbf{\pi}$ 和概率 $p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$ 重新组合成期望的形式 $\mathbb{E}_{\mathbf{\pi} \sim p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)}$，得：
$$
\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}\vert s) = \mathbb{E}_{\mathbf{\pi} \sim p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)}[L(\mathbf{\pi})\nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)]
$$

这一表达式**成功解决了之前的两个问题**。

首先，解决了梯度项复杂的问题。我们将对概率的求导，转化成了对对数概率的求导——在对数下，原先的乘积求导变为了加法求导：
$$
\nabla_{\mathbf{\theta}}\log p_{\mathbf{\theta}}(\pi|s) = \nabla_{\mathbf{\theta}}\sum_{t=1}^{n} \log p_{\theta}(\pi_t | s, \mathbf{\pi}_{1:t-1})
$$
对一个和求导，远比对一个乘积求导要简单得多。

其次，解决了不能采样近似的问题。由于含有概率 $p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$，我们将梯度转化为了期望形式，可通过**蒙特卡洛采样近似**，即从策略网络 $p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)$ 中采样出一批路径，然后计算平均值即可：
$$
\nabla_{\mathbf{\theta}} \mathcal{L}(\theta|s) = \mathbb{E}_{\mathbf{\pi} \sim p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)}[L(\mathbf{\pi})\nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)]\approx \frac{1}{N} \sum_{i=1}^{N} L(\mathbf{\pi}_i) \nabla_{\mathbf{\theta}} \log p_{\mathbf{\theta}}(\mathbf{\pi}_i|s)
$$
这就是著名的 REINFORCE 算法的核心。

### 4. 改进 - 引入 Baseline

REINFORCE 算法虽然可行，但存在一个巨大的问题：**梯度估计的方差太大**。首先，由于奖励恒为负，导致算法无法区分“好”和“没那么好”的行为，所有行为都被抑制，学习效率极低。其次，奖励值本身的大小（而非相对好坏）直接缩放梯度，导致梯度估计的方差巨大，使训练过程极不稳定且收敛缓慢。

解决方案是**引入一个基线** (Baseline) $b(s)$。这个基线的值与输入 $s$ 有关，但与具体采样的路径 $\mathbf{\pi}$ 无关。定义**优势函数**：
$$
A(\mathbf{\pi}, s) = L(\mathbf{\pi}) - b(s)
$$
我们用这个优势函数替换原始的 $L(\mathbf{\pi})$，得到新的梯度估计：
$$
\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}\vert s) = \mathbb{E}_{\mathbf{\pi} \sim p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)}[(L(\mathbf{\pi}) - b(s))\nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)]
$$
这就是论文[1]中的 Equation(9)。

但我们还需要证明，**引入这个基线不会改变梯度的期望值**（即梯度是无偏的），也即要证明减去的项的期望为零。
$$
\begin{aligned}
&\mathbb{E}_{\mathbf{\pi} \sim p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)}[b(s)\nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)] \\
= \enspace& \sum_{\mathbf{\pi}} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)b(s)\nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) \\
= \enspace& b(s)\sum_{\mathbf{\pi}} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s)\nabla_\mathbf{\theta} \log p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) \\
= \enspace& b(s)\sum_{\mathbf{\pi}} \nabla_\mathbf{\theta} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) \\
= \enspace& b(s)\nabla_\mathbf{\theta}\sum_{\mathbf{\pi}} p_{\mathbf{\theta}}(\mathbf{\pi} \vert s) \\
= \enspace& b(s)\nabla_\mathbf{\theta}(1) \\
= \enspace& 0 \\
\end{aligned}
$$
我们从梯度中减去了一个期望为零的项，所以梯度本身是无偏的。

引入 Baseline 后，如果 $L(\mathbf{\pi}) > b(s)$（当前路径比基准差），则优势为正，我们会降低这条路径的概率；如果 $L(\mathbf{\pi}) < b(s)$（当前路径比基准好），则优势为负。但由于我们的目标是最小化成本，梯度下降的更新方向是梯度的负方向，- (负优势 * 梯度项) 最终会提升这条好路径的概率。

通过**选择一个好的基线**，我们可以让梯度围绕 $0$ 波动，**大大降低方差**，使训练更稳定、更快速。论文[1]中使用的基线 $b(s)$ 就是一个**由过去最好的策略网络进行贪婪解码**得到的成本，这是一个非常强且有效的基线。

![REINFORCE with Rollout Baseline 完整流程](REINFORCE.png)

### 5. 代码实现

有了完整的 REINFORCE with Baseline 算法，我们可以进行代码实现。

```python
# attention/model.py
import copy
import numpy as np
import torch
from torch import Tensor
from ml4co_kit import BaseModel, TSPSolver
from .env import AttentionEnv
from .encoder import AttentionEncoder
from .decoder import AttentionDecoder
from .policy import AttentionPolicy
    
    
class AttentionModel(BaseModel):
    def __init__(
        self, 
        env: AttentionEnv,
        encoder: AttentionEncoder,
        decoder: AttentionDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        super(AttentionModel, self).__init__(
            env=env,
            # The main model to be trained
            model=AttentionPolicy(
                env=env,
                encoder=encoder,
                decoder=decoder,
            ),
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.to(self.env.device)
        
        # Create a separate baseline model
        baseline_encoder = copy.deepcopy(encoder)
        baseline_decoder = copy.deepcopy(decoder)
        self.baseline_model = AttentionPolicy(
            env=env,
            encoder=baseline_encoder,
            decoder=baseline_decoder,
        ).to(self.env.device)
        self.baseline_model.eval()  # Set to evaluation mode permanently
        self.update_baseline()  # Initialize baseline with policy weights
        
        # Store validation metrics
        self.val_metrics = []
        
    def update_baseline(self):
        """Copies the weights from the policy model to the baseline model."""
        self.baseline_model.load_state_dict(self.model.state_dict())
        
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training, validation, and testing.
        """
        self.env.mode = phase
        # unpack batch data
        points, ref_tours = batch
        # points: (batch_size, num_nodes, 2)
        # ref_tours: (batch_size, num_nodes + 1)
        if phase == "train":
            # --- 1. Policy Rollout (stochastic) ---
            # Gradients are tracked for this rollout.
            self.model.train() # Ensure model is in training mode
            reward, sum_log_probs, tours = self.model(points, mode='sampling')
            policy_cost = -reward  # Reward is negative tour length
        elif phase == "val":
            with torch.no_grad():
                self.model.eval() # Set model to evaluation mode
                # Evaluate the policy model
                reward, sum_log_probs, tours = self.model(points, mode='greedy')
                policy_cost = -reward
            
        # --- 2. Baseline Rollout (greedy) ---
        # No gradients are needed for the baseline.
        with torch.no_grad():
            reward, _, baseline_tours = self.baseline_model(points, mode='greedy')
            baseline_cost = -reward  # Reward is negative tour length
                
        # --- 3. Calculate REINFORCE Loss ---
        # The advantage is the gap between the sampled solution and the greedy baseline.
        advantage = policy_cost - baseline_cost
        # The loss is the mean of advantage-weighted negative log-probabilities.
        loss = (advantage * sum_log_probs).mean()

        if phase == "val":
            # Evaluate the tours
            costs_avg, _, gap_avg, _ = self.evaluate(points, tours, ref_tours)
            baseline_costs_avg, _, _, _ = self.evaluate(points, baseline_tours, ref_tours)

        # --- 4. Logging ---
        metrics = {f"{phase}/loss": loss}
        # print(f"loss: {loss.item()}")
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg, "val/gap_avg": gap_avg, "val/baseline_costs_avg": baseline_costs_avg})
            self.val_metrics.append(metrics)
        for k, v in metrics.items():
            self.log(k, float(v), prog_bar=True, on_epoch=True, sync_dist=True)
        # return
        return loss if phase == "train" else metrics   
    
    def on_validation_epoch_end(self):
        # Aggregate the costs from all validation batches
        avg_policy_cost = np.array([x['val/costs_avg'] for x in self.val_metrics]).mean()
        avg_baseline_cost = np.array([x['val/baseline_costs_avg'] for x in self.val_metrics]).mean()
        # Baseline Update
        if avg_policy_cost < avg_baseline_cost:
            self.update_baseline()
        self.val_metrics.clear()  # Clear the metrics for the next epoch

    def evaluate(self, x: Tensor, tours: Tensor, ref_tours: Tensor):
        """
        Evaluate the model's performance on a given set of tours.
        
        Args:
            x: (batch_size, num_nodes, 2) tensor representing node coordinates.
            tours: (batch_size, num_nodes+1) tensor representing predicted tours.
            ref_tours: (batch_size, num_nodes+1) tensor representing reference tours.
        
        Returns:
            costs_avg: Average cost of the predicted tours.
            ref_costs_avg: Average cost of the reference tours.
            gap_avg: Average gap between predicted and reference tours.
            gap_std: Standard deviation of the gap.
        """
        x = x.cpu().numpy()
        tours = tours.cpu().numpy()
        ref_tours = ref_tours.cpu().numpy()
            
        solver = TSPSolver()
        solver.from_data(points=x, tours=tours, ref=False)
        solver.from_data(tours=ref_tours, ref=True)
        costs_avg, ref_costs_avg, gap_avg, gap_std = solver.evaluate(calculate_gap=True)
        return costs_avg, ref_costs_avg, gap_avg, gap_std
```

注意，`on_validation_epoch_end` 是 Pytorch Lightning 的一个钩子函数，会**在每个 validation 的 epoch 结尾处被调用**。这里，我们对原文[1]的方法作了适当的简化，即在当前模型的平均路径长度小于 Baseline 模型时，就更新 Baseline 模型。原文在此处用了T-检验的方法判断模型显著更好。

## 训练过程

最后，我们对模型进行训练和验证。

```python
# attention/train.py
from .env import AttentionEnv
from .encoder import AttentionEncoder
from .decoder import AttentionDecoder
from .model import AttentionModel
from ml4co_kit import Trainer


if __name__ == "__main__":
    model = AttentionModel(
        env=AttentionEnv(
            mode="train",
            train_batch_size=32,
            val_batch_size=4,
            train_path="data/tsp20/tsp20_gaussian_train.txt",
            val_path="data/tsp20/tsp20_gaussian_val.txt",
            device="cuda",
        ),
        encoder=AttentionEncoder(
            embed_dim=128,
            num_heads=8,
            hidden_dim=512,
            num_layers=3,
        ),
        decoder=AttentionDecoder(
            embed_dim=128,
            num_heads=8,
        ),
    )
    
    trainer = Trainer(model=model, devices=[0], max_epochs=20)
    trainer.model_train()
```

运行该 Python 文件，即可在 Wandb 中看到我们的实验结果。

![](result.png)

可以看到，经过大约 5 个 epoch，`val/loss` 收敛在 0 左右；经过大约 16 个 epoch，`val/gap_avg` 收敛在 8.5% 左右。

至此，ML4CO 的第二种范式（Transformer 自回归 + 强化学习）实现完毕。这是目前 ML4CO 中**强化学习**模型的基础。

## 参考资料
[1] W. Kool, H. Van Hoof, and M. Welling, “Attention, learn to solve routing problems!” *arXiv preprint arXiv*:1803.08475, 2018.
[2] [https://github.com/Thinklab-SJTU/ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)
[3] [https://github.com/Thinklab-SJTU/ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)