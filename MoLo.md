# MoLo: Motion-augmented Long-short Contrastive Learning for Few-shot Action Recognition

> Wang, Xiang, et al. "MoLo: Motion-augmented Long-short Contrastive Learning for Few-shot Action Recognition." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

## 1 Motivation & Contribution

- 主流少样本动作识别的方法都是基于度量的元学习方法，通过将视频映射到适当的特征空间，然后计算对齐度量来预测查询集标签。这些方法依然有着两个局限性：
  - 基于对齐度量的方法都局限于局部的帧级对齐，而没有考虑到视频的全局时序信息，容易受不同类别的相似视频帧干扰。
  - 在当今主流的监督动作识别领域，运动信息是非常重要的，但是少样本动作识别都没有考虑到运动信息。

## 2 MoLo

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MoLo1.png)

### 2.1 Overall architecture

- MoLo总体上呈双流架构，分为长短时序对齐流和运动增强流，两个流输出的特征都会经过帧级对齐度量，然后将二者分数融合得到最后的预测结果。

### 2.2 Long-short Contrastive Objective

- 本文添加了一个可学习的全局token用于聚合全局时序信息，作者认为同一类的视频其全局时序特征应该与其对应同一类别的局部帧级特征在特征空间中更接近，因此作者采用了一个长短时序对比损失优化模型，提高局部帧级特征对全局上下文信息的感知能力。
- 全局特征和帧级特征，使用Temporal Transformer进行信息融合：
$$\tilde{f}_i=\text{Tformer}([f^{token},\text{GAP}(f_i^1,...,f_i^T)]+f_{pos})$$
- 帧级对齐度量：
$$D(\tilde{f}_i,\tilde{f}_q)=\mathcal{M}([\tilde{f}_i^1,...,\tilde{f}_i^T],[\tilde{f}_q^1,...,\tilde{f}_q^T])$$
- 长短对比损失函数：最大化所有正配对相似度分数的总和，并最小化所有负配对相似度分数的总和：
$$\begin{aligned} & \mathcal{L}_{LG}^{base}=-\log\frac{\sum_{i}sim(\tilde{f}_{q}^{token},\tilde{f}_{p}^{i})}{\sum_{i}sim(\tilde{f}_{q}^{token},\tilde{f}_{p}^{i})+\sum_{j\neq p}\sum_{i}sim(\tilde{f}_{q}^{token},\tilde{f}_{j}^{i})}\\  & -\log\frac{\sum_{i}\sin(\tilde{f}_{q}^{i},\tilde{f}_{p}^{token})}{\sum_{i}sim(\tilde{f}_{q}^{i},\tilde{f}_{p}^{token})+\sum_{j\neq q}\sum_{i}sim(\tilde{f}_{j}^{i},\tilde{f}_{p}^{token})}\end{aligned}$$

### 2.3 Motion Autodecoder

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MoLo2.png)

- 为了将运动信息纳入少样本动作识别中，作者通过STM中的帧级特征差值模块获取运动特征。

$$f_i^{'1},...,f_i^{'T-1}=\mathcal{F}(f_i^1,...,f_i^T)$$

- 为了生成更好的运动特征，作者借鉴了MAE，提出了一个运动自编码器，通过对帧级RGB差值像素进行重构，来监督运动特征的生成。
- 接下来的过程与2.2一致，只是将帧级特征替换为运动特征。

$$D(\tilde{f}_i^{\prime},\tilde{f}_q^{\prime})=\mathcal{M}([\tilde{f}_i^{\prime1},...,\tilde{f}_i^{\prime T-1}],[\tilde{f}_q^{\prime1},...,\tilde{f}_q^{\prime T-1}])$$

- 最终，query video 和 support video的距离度量则为两部分的加权和：

$$D_{i,q}=D(\tilde{f}_i,\tilde{f}_q)+\alpha D(\tilde{f}_i^{\prime},\tilde{f}_q^{\prime})$$

- 最终的损失函数则为三种任务损失函数的加权和：

$$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{1}(\mathcal{L}_{LG}^{base}+\mathcal{L}_{LG}^{motion})+\lambda_{2}\mathcal{L}_{Recons}$$

- 在few-shot inference 时仅根据加权的距离度量按最近邻原则进行分类，而无需重建运动解码器。

## 3 Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MoLo3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MoLo4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MoLo5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MoLo6.png)
