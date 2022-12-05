# Spatio-temporal Relation Modeling (STRM)

> Thatipelli A, Narayan S, Khan S, et al. Spatio-temporal relation modeling for few-shot action recognition[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 19958-19967.

## 一、小样本动作识别

1. 小样本：每个分类只有少量的已标注样本。

2. 相关名词：

   （1）支持集（S）：已标注的（少量）样本。

   （2）查询样本（Q）：未标注的样本。

3. 分类思路：用查询样本与支持集的每一个样本对比（距离）或者与每一类的中心对比（距离），类似于K=1的K近邻（但是高维视频不能像K近邻一样有简单的闵可夫斯基距离度量）。

## 二、Temporal-Relational CrossTransformers (TRX)

> Perrett T, Masullo A, Burghardt T, et al. Temporal-relational crosstransformers for few-shot action recognition[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021: 475-484.

（算是STRM的前置工作）

1. 特征提取：使用骨架网络将每一帧H\*W\*3的RGB图像转化为1\*D的特征向量。

2. 动作对：视频中的每两帧组成一组动作对（因为一般来说，动作都是画面的变化，一帧图像不能表示一个动作，所以想到用两帧来表示）。
   $$
   Q_p=[\Phi (q_{p1})+PE(q_{p1}),\Phi (q_{p2})+PE(q_{p2})]\in R^{2\times D}\\
   \Pi =\{(n1,n2)\in N^2:1\leq n_1<n_2\leq F\}
   $$
   其中Φ是特征提取网络，PE是位置编码。

   动作对的支持集：
   $$
   S^c=\{S_{km}^{c}:(1\leq k \leq K)\and{(m\in \Pi)}\}
   $$

3. 线性变换矩阵（作transformation，类似于attention中的Q/K/V）：

   Υ：query（2\*D=>dk），Γ：key（2\*D=>dk），Λ：value（2\*D=>dv）。

4. 交叉注意力：
   $$
   a_{kmp}^{c}=L(\Gamma\cdot S_{km}^{c})\cdot L(\Upsilon\cdot Q_p)\\
   \tilde{a}_{kmp}^{c}=\frac{exp(a_{kmp}^{c})/\sqrt{d_k}}{\sum_{l,n}exp(a_{lnp}^{c})/\sqrt{d_k}}(softmax标准化)
   $$

5. 查询样本第p个动作对对应支持集k类的样本原型：
   $$
   t_{p}^{c}=\sum_{km}\tilde{a}_{kmp}^{c}v_{km}^{c}\\
   v_{km}^{c}=\Lambda\cdot S_{km}^{c}(支持集作transformation得到的值向量)
   $$

6. 查询样本第p个动作对到支持集k类的距离：
   $$
   T(Q_p,S^c)=||t_p^c-u_p||\\
   u_p=\Lambda\cdot Q_p(查询样本作transformation得到的值向量)
   $$

7. 查询样本对到支持集k类的距离：求平均
   $$
   T(Q,S^c)=\frac{1}{|\Pi|}\sum_{p\in \Pi}T(Q_p,S^c)
   $$

8. 多帧动作：目前是只考虑了两帧的动作对，复杂动作可能需要多帧（ω帧）：
   $$
   T^{\Omega}(Q,S^c)=\sum_{\omega\in\Omega}T^{\omega}(Q^{\omega},S^{c\omega})
   $$

9. TRX的缺陷：（1）容易受到其它物体和背景的干扰。（2）每个基数（ω）需要单独做 CrossTranformer。

## 三、Spatio-temporal Relation Modeling (STRM)

1. 模块组成：

   （1）时空富集：包含PLE（patch-level的富集，空间富集）和FLE（frame-level的富集，空间富集）。

   （2）时间关系模型：TRM。

   （3）查询类相关性分类器：Query-class similarity classifier。

2. PLE：每一帧的特征图做self attention。
3. FLE：将特征图平均池化为1\*D后组合成L\*D（L帧的视频），输入一个全MLP网络。包含应用于D维度的MLP用于混合位置特征和一个跨D的MLP，用于混合空间信息。同时包含skip connection。
4. TRM：是Ω={2}的TRX。
5. Query-class similarity classifier：将PLE得到的H矩阵通过线性变化得到z向量，与查询集的z做相似度计算。

## 四、结果

Kinetics: 91.2%, SSv2: 70.2%, HMDB: 81.3%, UCF101: 98.1%。