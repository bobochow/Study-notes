# HyRSM

> Wang X, Zhang S, Qing Z, et al. Hybrid relation guided set matching for few-shot action recognition[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 19948-19957.

## 一、引言

1. 小样本学习的研究方向：

   （1）数据增强：增加训练样本数量、提高数据多样性。如空间变形、语义特征增强。

   （2）基于优化：通过**元学习模型**快速适应新任务。如基于 LSTM 的元学习器、学习高效模型初始化、学习随机梯度下降优化器等。

   （3）基于度量：通过“学习比较“。在通过欧氏距离、余弦相似度或可学习的非线性度量来学习特征空间并比较查询和支持图像。

2. 时间对齐策略：将不同时间段的数据进行对齐，以**解决样本量不足**、**类别分布不均**等问题。在小样本学习中，由于训练集数量较少，可能会存在类别分布不均、噪声干扰等问题，因此需要采用一些技术手段提高分类的准确性。

   时间对齐策略通常包括以下两种：

   （1）基于**距离**的时间对齐：这种方法是通过计算时间序列之间的距离来进行对齐。常用的距离计算方法包括欧氏距离、曼哈顿距离、动态时间规整（DTW）等。基于距离的时间对齐能够考虑到不同时间序列之间的相似度，但是需要时间序列之间的长度和采样率相同。

   （2）基于**特征**的时间对齐：这种方法是通过提取时间序列的特征来进行对齐。常用的特征提取方法包括小波变换、离散余弦变换、傅里叶变换等。基于特征的时间对齐能够在时间序列长度不同或采样率不同的情况下进行对齐，但是需要对时间序列进行预处理和特征提取。

3. 基于**时间对齐策略**学习每个视频判别特征的小样本学习方法的缺点：

   （1）学习**个别特征**而不考虑**整个任务**可能会丢失模式中最相关的信息。这些方法实际上假设学习到的表示对不同的情景任务同样有效，并为所有测试时间任务维护一组固定的视频特征。

   （2）这些对齐策略可能会在**未对齐**的情况下失败。如果时间序列数据存在**偏差**或不对齐，即使使用距离或特征对齐等技术进行对齐，也不能消除这些不对齐导致的误差。在这种情况下，需要先对数据进行预处理或对齐，再应用时间对齐策略进行分类。

   ![158](https://gitee.com/zhenyu-yang20/network-images/raw/master/158.png)

4. 研究贡献：

   （1）提出了一个新的混合关系模块来捕获情景任务内部和相互关系，为不同的任务产生特定于任务的表示。

   （2）进一步将查询支持视频对距离度量重新表述为集合匹配问题，并使用双向均值豪斯多夫度量（bidirectional Mean Hausdorff Metric），它对复杂的动作具有鲁棒性。

   （3）对六个具有挑战性的数据集进行了广泛的实验，以验证所提出的 HyRSM 实现了优于最先进方法的性能。

## 二、混合关系支持集匹配（HyRSM）

1. 模型框架：

   ![159](https://gitee.com/zhenyu-yang20/network-images/raw/master/159.png)

2. 混合关系模块：通过在情景任务中聚合**跨视频表示的语义信息** G 来改进特征 fi。
   $$
   \tilde{f}_i=\mathcal{H}\left(f_i, \mathcal{G}\right) ; f_i \in\left[F_s, f_q\right], \mathcal{G}=\left[F_s, f_q\right]
   $$
   为了计算效率，将混合关系模块 H 分解为内部关系模块 Ha 和相互关系模块 He。

   内部关系模块 Ha 的方法包含多头自注意力（MSA）、Transformer、Bi-LSTM、Bi-GRU。

   相互关系模块 He 是使用全样本集增强当前样本的特征，包含相关性计算与加权平均：
   $$
   f_i^e=\mathcal{H}_i^e\left(f_i^a, \mathcal{G}^a\right)=\sum_j^{\left|\mathcal{G}^a\right|}\left(\kappa\left(\psi\left(f_i^a\right), \psi\left(f_j^a\right)\right) * \psi\left(f_j^a\right)\right)
   $$

3. 集合匹配度量：将视频之间的距离测量重新表述为一个集合匹配问题，这对非对齐的复杂实例具有鲁棒性。

   使用双向均值豪斯多夫度量：
   $$
   \mathcal{D}_b=\frac{1}{N_i} \sum\left(\min _{\tilde{f}_q^b \in \tilde{f}_q}\left\|\tilde{f}_i^a-\tilde{f}_q^b\right\|\right)+\frac{1}{N_q} \sum\left(\min _{\tilde{f}_i \in \tilde{f}_i}\left\|\tilde{f}_q^b-\tilde{f}_i^a\right\|\right)
   $$

## 三、结果

![161](https://gitee.com/zhenyu-yang20/network-images/raw/master/161.png)