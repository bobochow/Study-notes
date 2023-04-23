# Revisiting Classifier: Transferring Vision-Language Models for Video Recognition

> Wu, Wenhao, Zhun Sun, and Wanli Ouyang. "Revisiting classifier: Transferring vision-language models for video recognition." Proceedings of the AAAI, Washington, DC, USA (2023): 7-8.

## 1 Motivation & Contribution

### 1.1 Motivation

- 之前的大部分工作将大规模语言视觉模型迁移到视频领域中时主要对image encoder部分做改进，对于text encoder 没有做深入的研究，大多数方法采用的是添加可学习的text prompt。
- 作者实验发现，具有相同动词的类别标签在用CLIP text encoder 提取的特征间具有显著的类间相关性。
- 实验还发现最后一层线性层$(f:d\to c)$输入的d维特征向量也可以揭示类别标签类间关系。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis1.png)

- 对于二者的相似性，作者认为样本中包含的语义信息确实与类间联系相关。因此，作者进一步研究优化模型最后的线性分类头，用预训练的多模态模型提取的语义信息替代了一般的随机初始化的线性分类头，为高效迁移学习生成更好的语义目标。

### 1.2 Contribution

- 提出了一种新的视觉语言大模型模型迁移学习方法，改进了一般模型最后一层的分类器，揭示了文本知识可以显著提高迁移学习的效果。

## 2 Method

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis2.png)

### 2.1 Revisiting of Previous Tuning Paradigms

- Standard Vision Transferring Paradigm

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis3.png)

- Vision-Language Learning Paradigm

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis4.png)

### 2.2 Revisiting the classifier for efficient tuning

- 将可学习的随机初始化线性投影矩阵$W$替换为预定义矩阵$\hat{W}$。$\hat{W}$不是优化目标，以保护文本知识不受小批量带来的随机性的干扰，并且在不同的初始化之间提供公平的比较。

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis5.png)

- $\Large\text{Randomized Matrix}$

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis6.png)

- $\Large\text{Randomized Orthogonal Matrix}$：先随机初始化矩阵，然后通过QR分解，得到行向量正交矩阵。

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis7.png)

- $\Large\text{Linear Discriminant Projection}$：线性判别分析是一种有监督的降维技术（PCA是无监督的降维），其核心是使投影后类内方差最小，类间方差最大。
- 优点是计算速度快且能充分利用先验知识，缺点是它在很大程度上取决于用于计算投影矩阵的数据。当数据有限时，估计的相关性会有偏差。

    > [线性判别分析LDA原理总结 - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/6244265.html)

- $\text{\Large Textual Embedding Vectors}$：投影权重由标签的嵌入文本特征向量组成。
- 实验了两种文本编码器，一种为多模态对比学习模型中的文本编码器，另一种为掩码语言建模模型中的文本编码器。

![8](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis8.png)

## 3 Experiment

- Training & Inference：在提出的范式中，分类器从类名的文本嵌入中初始化，然后冻结，只留下视频编码器中的参数来学习。
- Limitation：性能受限于类别标签的表示方式，如果类别用数字表示则无法从text encoder中获得语义信息，而LDA则仍然能有帮助。

![9](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis9.png)

![10](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis10.png)

![11](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis11.png)

![12](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vsi12.png)

![13](https://raw.githubusercontent.com/bobochow/blog_img/main/img/text4vis13.png)
