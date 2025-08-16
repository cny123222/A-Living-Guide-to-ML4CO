# A Living Guide to ML4CO
A beginner's guide to Machine Learning for Combinatorial Optimization (ML4CO).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Blog Post](https://img.shields.io/badge/Blog-cny123222.github.io-brightgreen)](https://cny123222.github.io/)

![](ml4co.jpg)

## 👋 Introduction

I am currently working in the field of ML4CO, but I'm also a beginner in this area. I've found that this field is relatively niche, with limited systematic learning materials available.

Therefore, through this series, I hope to document my learning process, my understanding of classic papers, and my hands-on coding practices, organizing the knowledge in a clearer and more systematic way.

This project aims to provide a roadmap and reference materials for other beginners interested in ML4CO, hoping to help you get started more quickly. My expertise is limited, so there may be errors or omissions in the content. I warmly welcome any feedback or corrections via **Issues** or **email**!

## 🗺️ Learning Roadmap

This guide will walk you through the world of ML4CO in the following order.

### 1. Fundamentals
Before we begin, we need to understand the basic goals, common problems, and experimental workflows in ML4CO.

- **1.1. [Goals and General Workflow of ML4CO](https://cny123222.github.io/2025/07/25/A-Living-Guide-to-ML4CO/)**
- **1.2. [Common Combinatorial Optimization Problems and Data Generation](https://cny123222.github.io/2025/07/28/Common-CO-Problems-in-ML4CO/)**
- **1.3. [Common Baseline Solvers](https://cny123222.github.io/2025/07/28/Traditional-Solver-Baselines-in-ML4CO/)**

### 2. Core Paradigms
Currently, there are two main paradigms for solving combinatorial optimization problems with machine learning.

#### 2.1. Paradigm 1: Supervised GNN + Decoding
The core idea of this approach is to **learn the structure of the solution**. It uses a Graph Neural Network (**GNN**) to learn the graph structure of a problem, predict which edges are part of the optimal solution, and finally generates the final solution through a decoding algorithm.

- **[Understanding Graph Neural Networks (GNNs) - An ML4CO Perspective](https://cny123222.github.io/2025/07/26/Understading-GNN-An-ML4CO-perspective/)**
- **[Paradigm 1: GNN + Decoding Code Implementation](https://cny123222.github.io/2025/07/27/Paradigm-1-Supervised-GNN-Decoding/)**

#### 2.2. Paradigm 2: Autoregressive Transformer + Reinforcement Learning
The core idea of this approach is to **learn a policy for constructing the solution**. It uses a **Transformer** as a policy network to construct a solution step-by-step, and uses **Reinforcement Learning (REINFORCE)** to learn and improve based on the final quality of the solution.

- **[Understanding Transformers - An ML4CO Perspective](https://cny123222.github.io/2025/07/26/Understading-Transformer-An-ML4CO-perspective/)**
- **[Paradigm 2: Autoregressive Transformer + RL Code Implementation](https://cny123222.github.io/2025/08/01/Paradigm-2-Autoregressive-Transformer-RL/)**


### 3. Classic Papers
We will dive deep into and analyze milestone papers in the field of ML4CO.

- **3.1. [GCN4TSP: An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem](https://cny123222.github.io/2025/07/30/Paper-Reading-1-GCN4TSP/)**
- **3.2. [AM: Attention, learn to solve routing problems!](https://cny123222.github.io/2025/07/30/Paper-Reading-2-AM/)**
- **3.3. (More to come...)**

### 4. Code Reading
Beyond theory and high-level implementation, we will also delve into official codebases to understand the brilliant details of their implementation.

- **4.1. (Coming soon...)**

### 5. Related Topics
Sharing some useful or interesting related topics encountered in research and practice.

- **5.1. [Pytorch Tensors: A Beginner's Guide](https://cny123222.github.io/2025/08/16/Pytorch-Tensors-A-Beginner-s-Guide/)**
- **5.2. [Fancy but Useful Tensor Operations](https://cny123222.github.io/2025/08/14/Fancy-but-Useful-Tensor-Operations/)**
- **5.3. (Feel free to suggest topics you're interested in via Issues!)**

---

## 🤝 How to Contribute & Get in Touch
- If you find any errors or have suggestions, please feel free to create an [**Issue**](https://github.com/cny123222/A-Living-Guide-to-ML4CO/issues)!
- If you want to fix an issue or add content directly, you are very welcome to submit a [**Pull Request**](https://github.com/your-username/A-Living-Guide-to-ML4CO/pulls).
- You can also contact me via email: _cny123222 AT sjtu.edu.cn_

## 🙏 Acknowledgements
The experimental parts of this tutorial series are primarily based on the [ML4CO-Kit framework by SJTU-Rethinklab](https://github.com/Thinklab-SJTU/ML4CO-Kit). Thanks to them for providing this convenient tool for the community.