# 两阶段动作对齐网络（TA2N）

> Li S, Liu H, Qian R, et al. TA2N: Two-stage action alignment network for few-shot action recognition[C]. Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(2): 1404-1411.

## 一、引言

1. 小样本学习的常用方法：度量学习、数据增强、基于优化的方法。

   度量学习存在的问题：不同的动作实例表现出不同时间分布导致查询和支持视频之间严重的错位问题，称为**动作持续时间错位**（AEM）。

2. 相关工作：

   （1）TARN：使用一个**逐段注意模块**执行文本对齐。

   （2）ARN：使用注意力机制来定位时间块的差异。

   （3）OTAM：使用动态**时间扭曲算法**的一种变体对齐。

3. 文章贡献：

   （1）提出了一种新颖的两阶段动作对齐网络（TA2N）对视频执行联合时空动作对齐。

   （2）大量实验表明，文章提出的方法可以消除错位，并在少镜头视频动作识别中取得最先进的结果。

## 二、TA2N

1. 模型结构：

   <img src="images/165.png" alt="165" style="zoom: 67%;" />

2. 时间转换模块（TTM）：缓解动作持续时间未对准问题。包含定位网络 L 和时间仿射变换 T。
   $$
   \hat{f_X}=\mathbf{T}_\phi\left(f_X\right), \phi=\mathbf{L}\left(f_X\right)\\
   \phi是生成的变形参数
   $$
   <img src="images/166.png" alt="166" style="zoom:67%;" />

3. 时间协调模块（TC）：TTM 只能处理线性的时间错位，对于非线性的时间错位，在时间维度上作通道注意力：

   <img src="images/167.png" alt="167" style="zoom: 33%;" />
   $$
   M=\operatorname{Softmax}\left(\frac{\left(W_k \cdot G\left(\hat{f}_s\right)\right)\left(W_q \cdot G\left(\hat{f}_q\right)\right)^T}{\sqrt{\operatorname{dim}}}\right)\\
   \tilde{f}_q=M \cdot\left(W_v \cdot G\left(\hat{f}_q\right)\right)
   $$

4. 空间协调模块（SC）：减少动作演变中的空间变化，如演员位置。包括两个步骤：轻量级偏移预测和偏移掩码生成。

   <img src="images/168.png" alt="168" style="zoom:60%;" />

## 三、结果

![169](images/169.png)