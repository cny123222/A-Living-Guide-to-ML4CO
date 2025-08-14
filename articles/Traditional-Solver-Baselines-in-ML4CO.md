---
title: Traditional Solver Baselines in ML4CO
date: 2025-07-28 10:44:58
index_img: img/ml4co.png
tags:
  - AI
  - ML4CO
category:
  - ML4CO
  - Basics
sticky: 300
---

本文将介绍 ML4CO 中常见的 baseline 求解器，如 Gurobi、Concorde、LKH-3、OR-Tools 等。

<!-- more -->

## 引言

在 ML4CO 实验中，我们经常需要 Baseline 求解器求解出问题的（也许是近似）精确解。这些求解器一般是传统求解器，运行速度慢但求解的效果较好。一方面，可以计算我们模型的求解质量与精确解之间的 gap，以评估我们模型的表现（这也是我们在论文的表格中经常看到的）；另一方面，基于监督学习的模型需要事先准备好问题的精确解，才能进行训练。

在这篇博客中，我们将对**常用的 Baseline 求解器及其使用范围**做简单的了解。这些求解器的具体使用都可以调用 [**ML4CO-Kit**](https://github.com/Thinklab-SJTU/ML4CO-Kit) 中的接口，但部分求解器（如 Gurobi）可能需要额外注册等，在此不再展开。

## 通用精确求解器

这类求解器不局限于求解某一特定问题，而是能够求解标准化的混合整数线性规划模型（MILP）。**凡是能表达为混合整数线性规划的问题**（如 **TSP、MIS、MCl、MCut、MVC、OP** 等），都可以使用这类求解器求解。

{% note info %}

混合整数线性规划模型（Mixed-Integer Linear Programming, MILP）指目标函数和约束条件均为线性，部分决策变量限制为整数的数学规划问题。

{% endnote %}

为了使用这类求解器，我们需要将一个具体的组合优化问题，通过定义变量、约束和目标函数，**建模成 MILP 的形式**。一旦建模完成，这些通用求解器就能接手，并运用其强大的内部算法（如分支定界、割平面等），**在有限时间内给出保证最优的解**。

### Gurobi

Gurobi[1] 是当今最强大、最受欢迎的商业数学优化求解器之一，能够高效求解**各类标准化的优化问题**，如：混合整数线性规划（MILP）、混合整数二次规划（MIQP）及线性规划（LP）、二次规划（QP）等。在 ML4CO 中，它常被用作 **TSP、MIS**、MCl、MCut、MVC、OP 等问题的求解器。

**使用方法**：使用 Gurobi 求解器时，需要先将问题建模成 MILP 模型，再通过 Gurobi 提供的 Python API（`gurobipy`）将定义好的变量、目标函数和约束条件输入到模型对象中，调用 `model.optimize()` 进行求解，最后从 Gurobi 返回的结果中读取解的内容。

需要注意，Gurobi 求解的**问题规模应在中小型范围内**（如 TSP 节点数不超过 500），大规模问题的求解速度会急剧下降。

## 特定问题精确求解器

对于一些极其著名和被广泛研究的问题，社区已经开发出了专门针对它们的、性能远超通用求解器的**专用求解器**。

### Concorde

Concorde[2] 是一个**专门用于求解旅行商问题 (TSP)** 的求解器，其内部算法都针对 TSP 问题的组合结构进行了高度优化。

**使用方法**：Concorde 通过命令行调用。它有自己标准的输入格式，即著名的 TSPLIB 格式。我们需要将 TSP 实例转换成这种 `.tsp` 文件格式，然后将其作为输入传给 Concorde 的可执行文件。当然，ML4CO-Kit 已经将调用过程进行了封装，我们可以直接使用。

Concorde 在求解 TSP 问题上，**其速度和能处理的问题规模远超 Gurobi 等通用求解器**。然而，Concorde **只能求解纯粹的 TSP 问题**，而无法求解 TSP 的诸多变体（如 ATSP、CVRP 等）。

## 启发式求解器

在很多场景下，我们并不强求得到最优解，而是希望快速得到一个足够好的近似解。启发式求解器不保证解的最优性，但它们通过精心设计的启发式规则，能在极短的时间内找到非常接近最优解的高质量解。

### LKH-3

LKH-3[3] 基于 Lin-Kernighan **启发式算法**，是一个先进的开源求解器。它利用**局部搜索**，从一个初始解出发，通过不断地进行复杂的**边交换**（k-opt），来持续迭代地改进当前解，直到无法进一步提升为止。LKH-3 可以**高效求解中大规模的 TSP 及其变体问题，如 ATSP、CVRP 等**。

**使用方法**：LKH-3 通过命令行调用。我们需要提供一个问题实例文件和一个参数文件，参数文件中定义了问题的类型、运行时间限制、迭代次数等。同样，ML4CO-Kit 等框架也为其提供了方便的 Python 接口。

LKH-3 **求解速度极快**（和精确求解器相比），解的质量较高，可用于求解大规模问题，但不保证解的最优性。

### KaMIS

KaMIS (Karlsruhe Maximum Independent Set Solver)[4] 源自德国卡尔斯鲁厄理工学院，是一个**专门用于求解 MIS 问题**的顶级求解器。KaMIS 并非依赖单一算法，而是巧妙地组合了图规约与核化、高级局部搜索等**多种技巧来高效地探索解空间**，能够在大规模的图上快速收敛到高质量的解。

### 经典启发式算法（Classic Heuristics）

一些更简单、更基础的启发式方法，如**最近邻算法 (Nearest Neighbour)**, **插入法 (Insertion Heuristics)**, 以及 **2-Opt** 等局部搜索算法，也常被作为比较模型性能的对象。

## 求解器套件

之前所说的都是单一求解器，那么能不能建造一个工具箱，为组合优化问题提供一站式解决方案呢？

### OR-Tools

OR-Tools[5] 是由 Google 开发和维护的一个开源的、用于解决组合优化问题的**软件套件**。它的设计初衷是让优化技术对广大开发者更加友好和易用，它提供了统一的 Python, C++, Java 和 C# 接口。

OR-Tools 内部包含了多个求解器组件，如：
- **CP-SAT 求解器**：一个基于约束规划 (Constraint Programming, CP) 和布尔可满足性 (SAT) 问题的求解器，擅长处理包含复杂逻辑约束、调度、分配和排序类的问题。在很多问题上，它都能给出保证最优的解。
- **路径规划库**：专门用于解决各类车辆路径问题 (VRP) 和旅行商问题 (TSP)。这个库的核心是高性能的启发式算法，如引导局部搜索 (Guided Local Search) 和禁忌搜索 (Tabu Search)，与 LKH-3 的定位非常相似。
- **线性规划与整数规划求解器**：OR-Tools 提供了统一的接口，可以调用多种第三方的线性规划和混合整数规划求解器，包括开源的 SCIP、GLOP，也包括商业的 Gurobi 和 CPLEX 。

**使用方法**：OR-Tools 的 API 设计非常高层次和声明式。我们不需要像使用 Gurobi 那样手动构建复杂的数学矩阵，而是更自然地描述问题的条件，很大程度上降低了建模的门槛。

## 参考资料
[1] L. Gurobi Optimization, “Gurobi optimizer reference manual (2020),” 2023.
[2] D. Applegate, R. Bixby, V. Chvatal, and W. Cook, “Concorde tsp solver,” 2006.
[3] K. Helsgaun, “An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems,” *Roskilde: Roskilde University*, vol. 12, 2017.
[4] S. Lamm, P. Sanders, C. Schulz, D. Strash, and R. F. Werneck, “Finding near-optimal independent sets at scale,” in *2016 Proceedings of the eighteenth workshop on algorithm engineering and experiments (ALENEX)*. SIAM, 2016, pp. 138–150.
[5] Laurent Perron and Vincent Furnon. OR-tools, 2022. Preprint.