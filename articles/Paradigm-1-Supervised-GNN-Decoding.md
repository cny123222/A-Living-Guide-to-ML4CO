---
title: 'Paradigm 1: Supervised GNN + Decoding'
date: 2025-07-27 17:31:37
index_img: img/ml4co.png
tags:
  - AI
  - ML4CO
category:
  - ML4CO
  - Basics
sticky: 300
---

本文将介绍 ML4CO 的第一种范式（GNN 监督学习 + 解码）的代码实现。代码文件可以在 [https://github.com/cny123222/A-Living-Guide-to-ML4CO](https://github.com/cny123222/A-Living-Guide-to-ML4CO) 中找到。

<!-- more -->

## 方法概述

在 [A Living Guide to ML4CO](https://cny123222.github.io/2025/07/25/A-Living-Guide-to-ML4CO) 中，我们提到过 ML4CO 的第一种常见范式：**GNN 监督学习 + 解码**。它通过学习问题和解的结构，尝试预测最优解。

我们还是以 TSP 为例，这种方法的大致过程是：

- **数据准备**：既然是监督学习，我们需要给每个问题实例**准备已知最优解**。这些最优解来自传统求解器。
- **图的构建**：将每个问题实例构建成一张图，确定顶点和边的特征。
- **模型构建**：GNN 输入顶点和边的特征，输出一张**热力图（heatmap）**，每个元素 `P(i, j)` 代表边 `(i, j)` 属于最优解的概率。
- **训练过程**：GNN 读入问题实例，输出热力图，**与真实标签比较**，计算损失并更新。
- **解码过程**：以热力图为指导构建最终解，一般采用贪心搜索或束搜索。

其中，“**模型构建**”部分由于涉及 GNN 的知识，我们将其放在了 [Understading GNN - An ML4CO perspective](https://cny123222.github.io/2025/07/26/Understading-GNN-An-ML4CO-perspective/) 这篇博客中。我们用 Pytorch 搭建了一个包含 Embedding 层、图卷积层和输出层的简单 **Encoder 网络**。

这篇博客中，我们将直接使用上述 Encoder 网络，并逐一实现**数据准备、图的构建、训练过程、解码过程**这几部分。

{% note warning %}

注意，这里只使用**最简单的模型和最一般的方法**进行演示。效果更好的经典模型及方法见 ML4CO Paper Reading 系列，如 [Paper Reading #1: GCN4TSP](https://cny123222.github.io/2025/07/30/Paper-Reading-1-GCN4TSP/)。

{% endnote %}

## 数据准备

我们在 [Common CO Problems in ML4CO](https://cny123222.github.io/2025/07/28/Common-CO-Problems-in-ML4CO/) 中讲过 TSP **数据集的生成算法**，也在 [Traditional Solver Baselines in ML4CO](https://cny123222.github.io/2025/07/28/Traditional-Solver-Baselines-in-ML4CO/) 中介绍过 TSP 常用的 **baseline 求解器**。

在这里我们设置**顶点数为 20、数据分布为 Gaussian、求解器为 LKH**。使用 **ML4CO-Kit** 生成数据集的代码如下：

```python
# gnn/data_generator.py
from ml4co_kit import TSPDataGenerator

# initialization
tsp_data_concorde = TSPDataGenerator(
    num_threads=8,
    nodes_num=20,
    data_type="gaussian",
    solver="LKH",
    train_samples_num=1280,
    val_samples_num=128,
    test_samples_num=128,
    save_path="data/tsp20"
)

# generate
tsp_data_concorde.generate()
```

生成的训练集、验证集和测试集位于 `data/tsp20` 路径下，格式为 `.txt`，且均有 LKH 求解器生成的最优解。具体地说，每一行包含一个 TSP 实例，其**格式**为： 40 个 [0,1] 的浮点数（表示 20 个顶点的坐标）、`output`、21 个整数（表示最优路径依次访问的顶点编号，含终点）。

## 图的构建

这部分涉及**数据集的读入（Dataset）和加载（DataLoader）**。为了简单起见，我们选择继承 Pytorch 的 `Dataset` 类并直接在 `__getitem__` 方法中进行数据的处理。

{% note warning %}

请注意，这**并不是一种很好的处理方式**。首先，`Dataset` 的 `__init__` 函数会一次性将所有数据读入内存，在数据集过大的情况下会造成问题。其次，`__getitem__` 方法每次只处理单个元素，无法进行批处理，效率很低。此处使用只是比较直观清晰，更实际的实现可以参考附录[2]中的代码。

{% endnote %}

对于 TSP 来说，我们构建的图会是一个完全图，即任意两点之间都有双向的有向边。下面是简单的代码实现：

```python
# gnn/env.py
import numpy as np
import torch
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
        
    def __getitem__(self, index):
        points = self.points[index]  # shape: (V, 2)
        tour = self.tours[index]     # shape: (V+1,)

        node_num = points.shape[0]
        # create edge index
        src, dst = np.meshgrid(np.arange(node_num), np.arange(node_num))
        mask = (src != dst)
        src, dst = src[mask], dst[mask]  # shape: (E,)
        edge_index = np.stack([src, dst], axis=0)  # shape: (2, E)

        # calculate each edge's length
        edges = np.linalg.norm(points[src] - points[dst], axis=1)  # shape: (E,)
                
        # generate the ground truth
        gt_adj = np.zeros((node_num, node_num), dtype=bool)
        gt_adj[tour[:-1], tour[1:]] = True
        gt_adj = gt_adj | gt_adj.T  # make it undirected
        ground_truth = gt_adj[src, dst]
        
        # convert into tensors
        points = torch.tensor(points, dtype=torch.float32)  # shape: (V, 2)
        edges = torch.tensor(edges, dtype=torch.float32)  # shape: (E,)
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # shape: (2, E)
        tour = torch.tensor(tour, dtype=torch.long)  # shape: (V+1,)
        ground_truth = torch.tensor(ground_truth, dtype=torch.long) # shape: (E,)
        
        return points, edges, edge_index, ground_truth, tour
    
    def __len__(self):
        return self.points.shape[0]  # number of samples
```

我们创造了一个 `TSPDataset` 类，它有两个方法：
- `__init__` 方法：完成了数据集从文件中的读取，将点的坐标和参考路径保存。
- `__getitem__` 方法：返回任意一条数据。数据格式按照我们所写 `Encoder` 的要求构造，返回了 `points`、`edges`、`edge_index`、`ground_truth` 和 `tour`。前三个变量的含义我们在 `Encoder` 搭建的[博客](https://cny123222.github.io/2025/07/26/Understading-GNN-An-ML4CO-perspective/) 中已经讲过。 `tour` 是参考路径，`ground_truth` 是 heatmap 的目标，即出现在 `tour` 中的边为 1，其余边为 0，用于计算损失。

{% note info %}

这里的 `__getitem__` 函数中，用到了几处 Numpy 语法，可能较为难以理解，大家可以参考 Numpy 的官方文档[3]。但这并不是我们的重点，只需要理解 `__getitem__` 方法的返回值含义即可。

{% endnote %}

为了更好地使用 ML4CO-Kit 框架，我们还需要完成一个 `GNNEnv`，其作用与 `DataLoader` 类似。我们简单地完成 `load_data`、`train_dataloader`、`val_dataloader` 和 `test_dataloader` 四个方法，进行数据加载，因为我们的 `GNNModel` 会默认调用这些函数。

```python
# gnn/env.py
from ml4co_kit import BaseEnv
from torch.utils.data import DataLoader
    
    
class GNNEnv(BaseEnv):
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
        super(GNNEnv, self).__init__(
            name="GNNEnv",
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
        
    def load_data(self):
        self.train_dataset = TSPDataset(self.train_path) if self.train_path else None
        self.val_dataset = TSPDataset(self.val_path) if self.val_path else None
        self.test_dataset = TSPDataset(self.test_path) if self.test_path else None
        
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
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader
```

## 解码过程

在搭建完整的 `GNNModel` 之前，我们先完成最后一个模块—— `GNNDecoder` 的搭建。其任务是输入一张 heatmap，输出最终求解得到的路径。

由于我们之前一直对边采用**稀疏表示**，heatmap 的形状为 `(E,)`，此时为了解码的方便，需要将其**转换到稠密表示**，即邻接矩阵表示，形状为 `(V, V)`。代码中将直接采用 ML4CO-Kit 提供的 `np_sparse_to_dense` 函数。

这里，我们只实现最基础的贪心（Greedy）解码逻辑，即每次从剩余的顶点中加入当前概率最大的作为路径的下一个顶点。代码如下：

```python
# gnn/decoder.py
from torch import Tensor
from ml4co_kit import np_sparse_to_dense
import numpy as np

class GNNDecoder():
    def __init__(self, decoding_type: str = "greedy"):
        self.decoding_type = decoding_type
        
    def decode(self, heatmap: Tensor, nodes_num: int, edge_index: Tensor):
        """
        Args:
            heatmap: (B, E) tensor representing edges being selected
            nodes_num: int, number of nodes
            edge_index: (B, 2, E) Tensor with edges representing connections from source to target nodes
        Returns:
            tour: (B, V) tensor representing the tour
        """
        # Convert to numpy for processing
        heatmap = heatmap.cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        # Convert heatmap to a dense format
        batch_size = heatmap.shape[0]
        nodes_num = heatmap.shape[1]
        heatmap_dense = np.zeros((batch_size, nodes_num, nodes_num), dtype=np.float32)
        for idx in range(batch_size):
            heatmap_dense[idx] = np_sparse_to_dense(
                nodes_num=nodes_num, edge_index=edge_index[idx], edge_attr=heatmap[idx]
            )  # Convert into a real heatmap (V, V)
        # Decode the tour based on the heatmap
        if self.decoding_type == "greedy":
            return self._greedy_decode(heatmap_dense, batch_size, nodes_num)
        else:
            raise NotImplementedError(f"Decoding type '{self.decoding_type}' is not supported.")

    def _greedy_decode(self, heatmap: np.ndarray, batch_size: int, nodes_num: int):
        """
        Args:
            heatmap: (B, V, V) numpy array representing the heatmap
            batch_size: int, number of samples in the batch
            nodes_num: int, number of nodes
        Returns:
            tours: (B, V+1) numpy array representing the decoded tours
        """
        tours = []
        # Iterate over each instance
        for idx in range(batch_size):
            tour = []
            current = None
            for _ in range(nodes_num):
                if current is None:
                    # Start from the first node
                    next_node = 0
                else:
                    # Select the next node with the highest probability
                    next_node = np.argmax(heatmap[idx][current]).item()
                tour.append(next_node)
                heatmap[idx][:, next_node] = 0  # Remove the selected node
                current = next_node
            tour.append(0)  # Return to the starting node
            tours.append(np.array(tour))
        return np.array(tours)
```

## 训练过程

接下来，我们进入核心的训练部分，构建一个完整的 `GNNModel`。这里，我们选择继承自 ML4CO-Kit 的 `BaseModel`，只需要完成 `shared_step` 这一个函数。

在 `shared_step` 方法中，我们要分别写好训练（`phase == "train"`）和验证（`phase == "val"`）的完整逻辑。

```python
# gnn/model.py
import torch
from torch import nn, Tensor
from ml4co_kit import BaseModel, TSPSolver
from .env import GNNEnv
from .encoder import GCNEncoder
from .decoder import GNNDecoder

class GNNModel(BaseModel):
    def __init__(
        self,
        env: GNNEnv,
        encoder: GCNEncoder,
        decoder: GNNDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        super(GNNModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env = env
        self.model = encoder
        self.decoder = decoder
        self.to(self.env.device)
        
    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training, validation, and testing.
        """
        self.env.mode = phase
        # unpack batch data
        x, e, edge_index, ground_truth, ref_tour = batch
        # x: (B, V, H), e: (B, E, H)
        # edge_index: (B, 2, E), ground_truth: (B, E)
        # ref_tour: (B, V+1)
        e_pred = self.model(x, e, edge_index)  # shape: (B, E, 2)
        loss = nn.CrossEntropyLoss()(e_pred.view(-1, 2), ground_truth.view(-1))
        if phase == "val":
            e_prob = torch.softmax(e_pred, dim=-1) # shape: (B, E, 2)
            heatmap = e_prob[:, :, 1]  # shape: (B, E)
            tours = self.decoder.decode(heatmap, x.shape[1], edge_index)  # shape: (B, V+1)
            costs_avg, _, gap_avg, _ = self.evaluate(x, tours, ref_tour)
        # log
        metrics = {f"{phase}/loss": loss}
        # print(f"{phase} loss: {loss.item()}")
        if phase == "val":
            metrics.update({"val/costs_avg": costs_avg, "val/gap_avg": gap_avg})
        for k, v in metrics.items():
            self.log(k, float(v), prog_bar=True, on_epoch=True, sync_dist=True)
        # return
        return loss if phase == "train" else metrics   
            
    def evaluate(self, x: Tensor, tours: Tensor, ref_tour: Tensor):
        """
        Evaluate the model's performance on a given set of tours.
        
        Args:
            x: (B, V, H) tensor representing node features.
            tours: (B, V+1) tensor representing predicted tours.
            ref_tour: (B, V+1) tensor representing reference tours.
        
        Returns:
            costs_avg: Average cost of the predicted tours.
            ref_costs_avg: Average cost of the reference tours.
            gap_avg: Average gap between predicted and reference tours.
            gap_std: Standard deviation of the gap.
        """
        x = x.cpu().numpy()
        ref_tour = ref_tour.cpu().numpy()
            
        solver = TSPSolver()
        solver.from_data(points=x, tours=tours, ref=False)
        solver.from_data(tours=ref_tour, ref=True)
        costs_avg, ref_costs_avg, gap_avg, gap_std = solver.evaluate(calculate_gap=True)
        return costs_avg, ref_costs_avg, gap_avg, gap_std
```

`evaluate` 方法对得到的路径做评估，返回路径平均长度、参考路径平均长度、gap 的均值和标准差。其实现是调用了 ML4CO-Kit 的 `TSPSolver` 求解器的 `evaluate` 方法。

核心部分是 `shared_step` 方法。在这里，模型收到 dataloader 传来的一个 batch，将顶点和边的特征通过 Encoder，再将得到的结果与 `ground_truth` 计算损失。如果是验证，还会进行 tour 的 decoding 和评估。

## 实战训练及验证

最后，我们可以对模型进行训练。这里，我们使用 ML4CO-Kit 提供的 `Trainer`（它继承自 Pytorch Lightning 中的 `Trainer`）。

```python
# gnn/train.py
from .env import GNNEnv
from .encoder import GCNEncoder
from .decoder import GNNDecoder
from .model import GNNModel
from ml4co_kit import Trainer


if __name__ == "__main__":
    model = GNNModel(
        env=GNNEnv(
            mode="train",
            train_batch_size=32,
            val_batch_size=4,
            train_path="data/tsp20/tsp20_gaussian_train.txt",
            val_path="data/tsp20/tsp20_gaussian_val.txt",
            device="cuda",
        ),
        encoder=GCNEncoder(
            hidden_dim=64,
            gcn_layer_num=10,
            out_layer_num=3,
        ),
        decoder=GNNDecoder(
            decoding_type="greedy",
        ),
    )
    
    trainer = Trainer(model=model, devices=[0], max_epochs=20)
    trainer.model_train()
```

运行该 Python 文件，即可在 Wandb 中看到我们的实验结果。

![](img1.png)

可以看到，经过大约 10 个 epoch，`val/loss` 收敛在 0.13 左右，`val/gap_avg` 收敛在 9% 左右，模型效果比较一般。当然，因为这只是随手写的基础模型。

至此，ML4CO 的第一种范式（GNN 监督学习 + 解码）实现完毕。这是目前 ML4CO 中**监督学习**模型的基础。

## 参考资料
[1] [https://github.com/Thinklab-SJTU/ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit)
[2] [https://github.com/Thinklab-SJTU/ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101)
[3] [https://numpy.org/doc/stable/reference/index.html](https://numpy.org/doc/stable/reference/index.html)