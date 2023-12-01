# Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning

> Yao, Huanjin, Wenhao Wu, and Zhiheng Li. "Side4Video: Spatial-Temporal Side Network for Memory-Efficient Image-to-Video Transfer Learning." arXiv preprint arXiv:2311.15769 (2023).

## 1. Motivation

- 与DiST类似，都是基于Side-Tuning针对视频任务的改进，改进的关键是用于时间建模的旁支网络。

## 2. Method

![1](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/Side4Video1.png)

- 主要使用了一个时序卷积模块和CLS Token Shift MHSA，其中CLS Token Shift MHSA类似于将shifted CLS Token作为prompt。
- 最后使用全局平均池化将patch tokens转化为分类结果，取代了CLS Token。

- 主要工作：
  - Remove [CLS] token in side network
  - Feature fusion on patch tokens
  - Temporal module in Side4Video
  - [CLS] token shift self-attention

## 3. Experiment

![2](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/Side4Video2.png)

![3](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/Side4Video3.png)

![4](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/Side4Video4.png)
