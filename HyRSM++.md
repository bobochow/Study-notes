# HyRSM++: Hybrid Relation Guided Temporal Set Matching for Few-shot Action Recognition

> Wang, Xiang, et al. "HyRSM++: Hybrid relation guided temporal set matching for few-shot action recognition." arXiv preprint arXiv:2301.03330 (2023).

## 1. Motivation & Contribution

- Bi-MHM集合匹配度量并没有考虑时序信息，因此HyRSM++增加了一个时序相关性正则化，通过结合时间顺序信息进一步约束匹配过程。
- 此外，还将HyRSM++扩展到了半监督和无监督少样本动作识别任务中。

## 2. Method

### 2.1 Temporal Coherence Regularization

$$I(\tilde{f_i};\tilde{f_i^a},\tilde{f_i^b})=\begin{cases}\frac{1}{(a-b)^2+1}\cdot\left\|\tilde{f_i^a}-\tilde{f_i^b}\right\|,&\text{if}~|a-b|\leq\delta\\max(0,m_{ab}-\left\|\tilde{f_i^a}-\tilde{f_i^b}\right\|)&\text{if}~|a-b|>\delta\end{cases}$$

### 2.2 Semi-supervised Few-shot Action Recognition

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/HyRSMpp1.png)

### 2.3 Unsupervised Few-shot Action Recognition

- 首先采用现有的无监督学习方法来学习输入视频的初始化特征嵌入来生成少样本任务，然后利用深度聚类技术构建视频的伪类。根据聚类结果，我们能够通过对 N 路 K-shot 集进行采样来生成少样本任务

## 3. Experiment

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/HyRSMpp2.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/HyRSMpp4.png)

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/HyRSMpp3.png)
