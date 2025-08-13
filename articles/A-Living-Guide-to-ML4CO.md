---
title: A Living Guide to ML4CO
date: 2025-07-25 14:54:13
index_img: img/ml4co.png
tags:
  - AI
  - ML4CO
category:
  - ML4CO
sticky: 400
---

希望能成为一本给 Machine Learning for Combinatorial Optimization (ML4CO) 初学者的入门指南～（不断更新中...）

<!-- more -->

笔者正在做 ML4CO 方面的工作，但也刚刚入门。这个领域并不算热门，学习资料也不算多。我想能把我学习的过程、对论文的理解都记录下来，结合一些代码，整理得清晰一些，提升一下自己的水平，也希望给同样想做这个方向的同学们一些参考。

笔者水平非常有限，如果内容有误，欢迎交流指正！

## 1. ML4CO 的目标

组合优化问题（CO）主要分为两类，**选边**的问题（如旅行商问题 TSP）和**选点**的问题（如最大独立集 MIS），它们大多是 NP-hard 问题。因此，当问题的规模增大，传统求解器的求解时间急剧上升，变得不可接受。

我们考虑，是否能用机器学习（ML）来求解组合优化问题？机器学习也许可以学习解的结构，从而学会生成较为优秀的解，大幅降低求解时间。事实上，在实际应用中，我们并不需要得到最优解，一个与最优解较为接近的解也是可以接受的。

因此，ML4CO 的目标就是，在**尽可能短的时间**内，得到**尽可能高质量的解**。

## 2. 实验的一般步骤

ML4CO 实验的一般步骤包括：初始化求解器、读入数据集、求解、结果评估。

在这里，我们推荐使用 [**SJTU-Rethinklab 出品的 ML4CO-Kit 框架**](https://github.com/Thinklab-SJTU/ML4CO-Kit)进行实验。

```python
pip install ml4co_kit
```

以 TSP 问题为例，假设你已经写好了你的 TSP 求解器：

```python
from ml4co_kit import TSPSolver

def MyTSPSolver(TSPSolver):

    def __init__(self, **kwargs):
        # some code

    def solve(self, **kwargs):
        # some code
```

第一步，初始化求解器：

```python
solver = MyTSPSolver()
```

第二步，读入数据集：

```python
solver.from_txt("example/tsp50_guassian.txt", ref=True)
```
- `ref=True` 表示同时读入作为参考答案的路径（ref_tour）

第三步，求解：

```python
solver.solve()
```

第四步，结果评估：

```python
costs_avg, ref_costs_avg, gap_avg, gap_std = solver.evaluate(calculate_gap=True)
```
- `costs_avg` 表示求解路径的平均长度
- `ref_costs_avg` 表示参考路径的平均长度
- `gap_avg` 和 `gap_std` 是与参考路径长度偏差的平均值和标准差（均为百分比）

## 3. 常见的组合优化问题及数据生成

如前文所述，组合优化问题主要分为两类，分别是选边的（以 TSP 为代表）和选点的（以 MIS 为代表）。常见的组合优化问题及数据生成方法见我的博客 [Common CO Problems in ML4CO](https://cny123222.github.io/2025/07/28/Common-CO-Problems-in-ML4CO/)。

## 4. 常见的 baseline 求解器

在监督学习中，我们需要**提前获得实例的最优解**，这就需要用到传统求解器当作 baseline。常见 baseline 求解器的介绍见 [Traditional Solver Baselines in ML4CO](https://cny123222.github.io/2025/07/28/Traditional-Solver-Baselines-in-ML4CO/)。

## 5. 范式一：GNN 监督学习 + 解码

第一种范式使用监督学习，基本的想法是：**学习问题和解的结构**，从而预测最优解。其中骨干模型（backbone）一般采用图神经网络（**GNN**）。

我们以 TSP 为例，简单介绍这种方法：

- **数据准备**：既然是监督学习，我们需要给每个问题实例**准备已知最优解**。这些最优解来自我们之前说到的传统求解器。
- **图的构建**：将每个问题实例构建成一张图，确定顶点和边的特征。
- **模型构建**：GNN 输入顶点和边的特征，输出一张**热力图（heatmap）**，每个元素 `P(i, j)` 代表边 `(i, j)` 属于最优解的概率。
- **训练过程**：GNN 读入问题实例，输出热力图，**与真实标签比较**，计算损失并更新。
- **解码过程**：以热力图为指导构建最终解，一般采用贪心搜索或束搜索。

这种范式具体的**代码实现**见 [Understading GNN - An ML4CO perspective](https://cny123222.github.io/2025/07/26/Understading-GNN-An-ML4CO-perspective/) 和 [Paradigm 1: Supervised GNN + Decoding](https://cny123222.github.io/2025/07/27/Paradigm-1-Supervised-GNN-Decoding/) 这两篇博客。

## 6. 范式二：Transformer 自回归 + 强化学习

第二种范式使用强化学习，基本的想法是：像人一样**一步步构建解**，并从最终结果的好坏中学习。其中策略网络一般采用 **Transformer**，训练方法一般采用 **REINFORCE**。

我们仍以 TSP 为例，简单介绍这种方法：

- **数据准备**：强化学习的好处在于**不需要准备已知的最优解**。
- **模型构建**：策略网络一个节点一个节点地构建出最终的路径，在第 `t` 步时，
  - **输入**：Transformer 会接收**当前已经构建的部分路径以及所有节点的全局信息**作为输入
  - **输出**：模型输出是在所有尚未访问的节点中，**选择下一个要访问的节点的概率分布**
- **训练过程**：Transformer 构建出一条完整路径后，计算总长度，用**总长度作为奖励信号**，使用 REINFORCE 进行更新

这种范式具体的**代码实现**见 [Understading Transformer - An ML4CO perspective](https://cny123222.github.io/2025/07/26/Understading-GNN-An-ML4CO-perspective/) 和 [Paradigm 2: Autoregressive Transformer + RL](https://cny123222.github.io/2025/08/01/Paradigm-2-Autoregressive-Transformer-RL/) 这两篇博客。

## 7. 经典论文及方法分析

接下来，我们阅读一些 ML4CO 的重要论文。

1. GCN4TSP: [An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem](https://cny123222.github.io/2025/07/30/Paper-Reading-1-GCN4TSP/)

2. AM: [Attention, learn to solve routing problems!](https://cny123222.github.io/2025/07/30/Paper-Reading-2-AM/)

3. (持续更新中...)

## 参考资料
1. [https://github.com/Thinklab-SJTU/ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)