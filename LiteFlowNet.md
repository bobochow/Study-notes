# LiteFlowNet

> Hui T W, Tang X, Loy C C. Liteflownet: A lightweight convolutional neural network for optical flow estimation[C]. Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 8981-8989.

## 一、引言

1. LiteFlowNet 是轻量级的 FlowNet，160M的参数，比同年的 PWC-Net 还要小。

2. LiteFlowNet 的特点：

   （1）轻量的级联网络，提供金字塔级别的流处理。

   （2）早期校正和描述符匹配。

   （3）使用流正则化改善异常值和模糊边界。

3. 相关工作中的优点：

   （1）金字塔特征提取（不同尺度的运动信息）。

   （2）特征变形（wrapping）：使 I2 与 I1 的距离更近，网络学习更易。

   本文将 wrapping 与特征提取合并，更直接地缩短了特征空间的距离，而非原始RGB空间的距离。

## 二、LiteFlowNet

![74](images\74.png)

1. 总体架构：NetC 与 NetE：

   （1）NetC：维度收缩，将图像对转化为多尺度高维特征的两个金字塔。

   （2）NetE：维度扩展，级联流推理 + 流正则化。

2. 级联流推理：分为描述符匹配（M）和亚像素细化（S）两个部分。

   ![75](images\75.png)

3. 描述符匹配：

   （1）短程匹配：匹配每个维度下短距离的运动信息。

   （2）使用上一级的光流对 F2 作 wrapping，使 F2 与 F1在特征空间的距离更近。

4. 亚像素细化：用当前分辨率的流场再对 F2 作一次 wrapping，并再做一次流推理。将流场细化到亚像素精度，防止错误在上采样时被放大传递到下一个金字塔级别。

5. 流正则化：使用特征驱动的局部卷积（f-lcon），改善异常值（噪点）和模糊边界（边缘）。

   ![76](images\76.png)

6. 特征驱动的局部卷积：自适应构建每个位置不同的滤波器。

   （1）计算特征驱动的距离矩阵：
   $$
   D=R_D(F_1,\dot{x_s},O),其中O为亮度误差, O=||I_2(x+\dot{x})-I_1(x)||_2
   $$
   （2）根据 D 计算特征驱动滤波器（卷积核）g(x, y, c)。

## 三、结果

![77](images\77.png)

![78](images\78.png)