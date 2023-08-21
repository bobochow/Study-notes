# A Closer Look at Few-Shot Video Classification: A New Baseline and Benchmark

> Zhu, Zhenxi, et al. "A Closer Look at Few-Shot Video Classification: A New Baseline and Benchmark." (2021).

## 1. Motivation

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-baseline1.png)

- 当前主流的少样本动作识别方法大多忽视了视觉特征的表示，直接采用ImageNet预训练权重，但是ImageNet会包含相当一部分与少样本新类高度相关的类别，这破坏了少样本学习的基础假设，无法揭示方法的真实泛化性能。当前少样本动作识别缺乏从头训练的基准。
- 采用episode-based的元学习训练策略缺乏足够的训练样本用于从头训练，因此本文提出使用所有训练集样本训练一个特征提取器和线性分类头，在测试阶段则固定特征提取器的同时训练新的分类头，这样模型的特征表示得到了增强。

## 2. Method

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-baseline2.png)

### 2.1 Baseline

- Baseline在训练阶段使用所有的训练样本训练一个特征提取器和线性分类头，使用时序平均来进行时序融合，测试阶段固定特征提取器的同时基于元学习策略训练新的分类头。

### 2.2 Baseline Plus

- Baseline Plus则在Baseline的基础上增加了一些增强模型泛化能力的技巧。
- Dropout对于基于分类头的少样本识别能增强模型的返还能力，但无法用于基于对齐度量的方法。
- 在测试时还使用了weight imprinting，直接用训练样本得到的特征向量来作为线性分类器中该类别对应的权重。如果有超过1个的样本在新的类别中，通过求平均值的方法来设置新的权值。

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/weight-imprinting.png)

## 3. Experiments

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-baseline.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-baseline3.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-baseline4.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-baseline5.png)
