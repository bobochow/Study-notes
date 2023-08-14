# Learning Implicit Temporal Alignment for Few-shot Video Classification

> Zhang, Songyang, et al. “Learning Implicit Temporal Alignment for Few-Shot Video Classification.” Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021, <https://doi.org/10.24963/ijcai.2021/181>.

## 1. Motivation

- 一些通过显著记忆机制或时空注意力学习全局视频表示的方法缺少与时序对齐的结合，受到视频类内变化较大的影响，导致性能下降。
- 而一些基于显式时序对齐的方法往往需要计算密集的对齐矩阵，导致计算量大，且容易受到噪音的影响。
- 本文提出了一种隐式时序对齐方法，并与时空注意力结合，使得模型更加鲁棒且精确。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ITANet1.png)

### 2.1 Context Encoding Network

- 常规的时空注意力

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ITANet2.png)

### 2.2 Implicit Temporal Alignment Network

#### 2.2.1 Temporal Relation Encoding Module

- 首先使用平均池化压缩空间尺度，得到帧级特征；然后加入了位置编码，计算多头自注意力，用时间关系上下文增强每一帧的特征。

$$\begin{aligned}&\mathbf{Z}=\mathrm{Avg}\text{Pool}_{spatial}(\widetilde{\mathbf{F}})\\&\mathbf{Z}=\{\mathbf{z}_1,\cdots,\mathbf{z}_{n_t}\}\in\mathbb{R}^{n_t\times n_c}\end{aligned}$$

$$\begin{aligned}\mathbf{z}_{n_t}^p&=\mathbf{z}_{n_t}+\mathbf{p}_{n_t},\quad\mathbf{p}=f_{pos}(t)\in\mathbb{R}^{n_c}\\\mathbf{Z}^p&=\{\mathbf{z}_1^p,\cdots,\mathbf{z}_{n_t}^p\}\in\mathbb{R}^{n_t\times n_c}\end{aligned}$$

$$\widetilde{\mathbf{Z}}=\text{МultiНеаd}(\mathbf{Z}^p)+\mathbf{Z}^p$$

#### 2.2.2 Similarity Metric Module

- 接着计算经过自注意力后的特征间的相似性

$$S(\mathbf{X}_i,\mathbf{X}_j)=\frac1{n_t}\sum_{t=1}^{n_t}\frac{(\mathbf{\tilde{z}}_t^i)^\intercal\mathbf{\tilde{z}}_t^j}{||\mathbf{\tilde{z}}_t^i|||||\mathbf{\tilde{z}}_t^j||}$$

### 2.3 Meta Learning with Multi-task Loss

- 对每次元学习任务增加了视频分类辅助损失，直接在context encoding network后增加了一层线性层作为分类头。
- 最终的预测类别为：
$$\tilde{y}^q=\sum_{i=1}^{CK}S(\mathbf{X}_i^s,\mathbf{X}^q)y_i^s$$

$$\mathcal{L}_{all}=\mathcal{L}_{meta}+\beta\mathcal{L}_{sem}$$

$$\mathcal{L}_{sem}=\mathcal{L}_{CE}(\tilde{y}_{sem}^{s},y_{sem}^{s})+\mathcal{L}_{CE}(\tilde{y}_{sem}^{q},y_{sem}^{q})$$

## 3. Experiments

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ITANet3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ITANet4.png)
