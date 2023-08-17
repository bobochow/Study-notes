# Inductive and Transductive Few-Shot Video Classification via Appearance and Temporal Alignments

> Nguyen, Khoi D., et al. "Inductive and transductive few-shot video classification via appearance and temporal alignments." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

## 1. Motivation

- 早期的aggregation-based方法通过pooling、adaptive fusion、attention生成video-level的特征表示，但由于缺乏时序建模，效果较差。
- 而matching-based的方法显式地对齐两个视频帧序列，通过Dynamic Time Warping (DTW)  和 Optimal Transport (OT)等匹配方法，将两个视频间的距离作为匹配损失。采用DTW的方法匹配过于严格，易受噪声影响；而OT的方法需要迭代计算，计算开销大。
- 因此，本文并没有采用DTW计算序列相似度，而是与预设的时序匹配先验计算时序相似性，而提取的frame-level embedding则仅用于计算appearance相似性。从而以较低的计算代价，同时考虑appearance和temporal相似性。
- 此外，此前所有的少样本动作识别工作都属于归纳推理(Induction Inference)，即先从训练样本中学习得到一个通用的规则，再利用规则判断测试样本，这在处理小样本问题时会受训练样本数量的限制，导致性能不佳。而转导推理（Transductive Inference）是一种通过观察特定的训练样本，进而预测特定的测试样本的方法，能同时利用标记的训练数据和未标记的测试数据来提高性能，更加适用于小样本任务场景。本文提出了一种用于少样本视频动作识别的基于聚类的转导学习方法。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata1.png)

### 2.1 Appearance and Temporal Similarity Scores

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata.png)

#### 2.1.1 Appearance Similarity Score

- 对于appearance similarity，本文并没有和DTW一样严格限制顺序，而是所有帧与其对应最相似帧的相似性总和如图(a)所示，并用log-sum-exp近似max。

$$\mathbf{D}(i,j)=\frac{f_\theta(\mathbf{x}^i)^Tf_\theta(\mathbf{y}^j)}{||f_\theta(\mathbf{x}^i)||||f_\theta(\mathbf{y}^j)||}$$

$$\operatorname{sim}_a(f_\theta(\mathbf{X}),f_\theta(\mathbf{Y}))=\sum_{i=1}^M\max_j\mathbf{D}(i,j)\approx\sum_{i=1}^M\lambda\log\sum_{j=1}^M\exp^{\frac{\mathbf{D}(i,j)}\lambda}$$

#### 2.1.2 Temporal Similarity Score

- 对于temporal similarity，采用DTW的方法计算量较大，为了减少计算量，本文直接与预设的时序匹配先验矩阵计算时序相似性，如图(b)所示。该时序匹配先验矩阵源自理论上的两个视频完美匹配时的对角匹配矩阵，与OTAM中的Diagonal Matching Matrix相似。数学上，其沿垂直于对角线的任何直线的边缘分布是一个高斯分布，以对角线上的交点为中心。二者的区别在于Diagonal Matching Matrix对角元素是每列的均值。

$$\mathbf{T}(i,j)=\frac1{\sigma\sqrt{2\pi}}\exp^{-\frac{l^2(i,j)}{2\sigma^2}},\quad l(i,j)=\frac{|i-j|}{\sqrt{2}},$$

- 接着计算归一化后的 appearance similarity matrix 和 Temporal Order-Preserving Prior之间的KL散度，作为temporal similarity score。

$$\tilde{\mathbf{D}}(i,j)=\frac{\exp^{\mathbf{D}(i,j)}}{\sum_{k=1}^M\exp^{\mathbf{D}(i,k)}},\quad\tilde{\mathbf{T}}(i,j)=\frac{\mathbf{T}(i,j)}{\sum_{k=1}^M\mathbf{T}(i,k)}.$$

$$\operatorname{sim}_t(f_\theta(\mathbf{X}),f_\theta(\mathbf{Y}))=-KL(\tilde{\mathbf{D}}||\tilde{\mathbf{T}})=-\frac{1}{M}\sum_{i=1}^M\sum_{j=1}^M\tilde{\mathbf{D}}(i,j)\operatorname{log}\frac{\tilde{\mathbf{D}}(i,j)}{\tilde{\mathbf{T}}(i,j)}.$$

### 2.2 Training and Testing

- **训练**与基于原型的方法类似，

$$\mathcal{L}_{Sup}=-\mathbb{E}_{(\mathbf{X},y)\sim\mathcal{D}_b}\log\frac{\exp^{\operatorname{sim}_a(f_\theta(\mathbf{X}),\mathbf{W}_y)}}{\sum_{p=1}^{N_b}\exp^{\operatorname{sim}_a(f_\theta(\mathbf{X}),\mathbf{W}_p)}}-\alpha\mathbb{E}_{(\mathbf{X},y)\sim\mathcal{D}_b}\operatorname{sim}_t(f_\theta(\mathbf{X}),\mathbf{W}_y)$$

- 时序先验矩阵仅对SthV2有用，而对Kinetics效果不明显，因此在Kinetics上不使用时序先验矩阵(i.e., α = 0)。
- 此外，仅使用上述损失函数会导致模型仅学习每个类别的判别特征，泛化能力不足，因此添加了额外的损失函数：

$$\mathcal{L}_{Info}=-\frac1M\sum_{i=1}^M\sum_{j=1}^M\tilde{\mathbf{D}}(i,j)\log\tilde{\mathbf{D}}(i,j).$$

$$\mathcal{L}=\mathcal{L}_{Sup}+\nu\mathcal{L}_{Info}.$$

- **测试时**，预测分布如下：

$$p(c|\mathbf{X},\mathbf{W})=(1-\beta)\frac{\exp^{\operatorname{sim}_a(f_\theta(\mathbf{X}),\mathbf{W}_c)}}{\sum_{j=1}^N\exp^{\operatorname{sim}_a(f_\theta(\mathbf{X}),\mathbf{W}_j)}}+\beta\frac{\exp^{\operatorname{sim}_t(f_\theta(\mathbf{X}),\mathbf{W}_c)}}{\sum_{j=1}^N\exp^{\operatorname{sim}_t(f_\theta(\mathbf{X}),\mathbf{W}_j)}}$$

### 2.3 Prototype Refinement

- 在用于分类之前，可以使用支持样本(在归纳设置中)或同时使用支持和查询样本(在转导设置中)进一步改进原型。
- **Inductive Setting**：在归纳设置中，可以在支持样本上进一步微调原型

$$\mathcal{L}_{inductive}=-\mathbb{E}_{(\mathbf{X},y)\sim\mathcal{D}^s}\log\frac{\exp^{\operatorname{sim}_a(f_\theta(\mathbf{X}),\mathbf{W}_y)}}{\sum_{i=1}^N\exp^{\operatorname{sim}_a(f_\theta(\mathbf{X}),\mathbf{W}_i)}}$$

- **Transductive Setting**：转到推理通常采用具有新的分配函数的soft K-means形式。对于支持样本，其分配函数$z(\mathbf{X},c)$为独热标签向量，对于查询样本，其分配函数$z(\mathbf{X},c)$为预测分布。
- 每次更新迭代时，采用支持样本和查询样本的加权和更新原型，权重如下：

$$\mathbf{W}_c=\frac{\sum_{\mathbf{X}\sim\mathcal{S}}f_\theta(\mathbf{X})z(\mathbf{X},c)+\sum_{\mathbf{X}\sim\mathcal{Q}}f_\theta(\mathbf{X})z(\mathbf{X},c)}{\sum_{\mathbf{X}\sim\mathcal{S}}z(\mathbf{X},c)+\sum_{\mathbf{X}\sim\mathcal{Q}}z(\mathbf{X},c)},$$

## 3. Experiments

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata2.png)

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata6.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/fsvc-ata7.png)
