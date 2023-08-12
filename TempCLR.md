# TempCLR: Temporal Alignment Representation with Contrastive Learning

> Yang, Yuncong, et al. "TempCLR: Temporal Alignment Representation with Contrastive Learning." arXiv preprint arXiv:2212.13738 (2022).

## 1. Motivation & Contribution

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/TempCLR1.png)

- 现有的video-text matching方法大多是基于sentence-clip pairs，整个paragraph-video的对齐是隐式的，全局时序信息有所欠缺，并且易受噪声的影响。
- 因此，本文提出了一种基于序列的显式全局对齐方式，通过设置乱序对比学习代理任务，学习到了更好的全局时序表示。

## 2. Temporal Alignment Representation With Contrastive Learning

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/TempCLR2.png)

- 对于一个视频来说，都可被分为多个segment，并且每个segment都有一个对应的文本描述，而每个segment都由若干个clip组成。以往的基于clip-level对齐的工作往往会由于相似的clip被分配到不同的segment中而导致对齐的不准确，因此本文提出了一种基于sequence-level的对齐方式，即将每个sequence作为一个对齐单元，从而避免了clip-level对齐的不准确性。

### 2.1 Negative Sampling based on Temporal Granularity

- 本文采用对比学习的方法来学习全局时序表示，因此需要设计一个合适的负样本采样策略。
- 以往也有设置乱序帧为代理任务的对比学习工作，但是其是随机乱序，且缺少了文本信息。
- 本文提出了一种基于不同时序粒度的乱序方法，即首先在segment-level上进行乱序，然后在clip-level上进行乱序，从而得到了更加丰富的负样本。
- 此外，对比是 video-text 之间的，text和video是同一个instance的不同视角。

### 2.2 Contrastive learning

- minimize InfoNCE loss：

$$\mathcal{L}_{seq}(\mathbf{S}_a,\mathbf{S}_p,\mathcal{S}_n)=-\log\frac{\exp(d_{\{\mathbf{S}_a,\mathbf{S}_p\}}/\tau)}{exp(d_{\{\mathbf{S}_a,\mathbf{S}_p\}}/\tau)+\sum_{\mathbf{S}_n\in\mathcal{S}_n}\exp(d_{\{\mathbf{S}_a,\mathbf{S}_n\}}/\tau)}$$

### 2.3 Sequence-level distance

- 本文也没有放弃一般的clip-level对齐，采用了DTW和OTAM。

$$C(n_a,n_p)=D(n_a,n_p)+\min\{C(n_a-1,n_p-1),C(n_a-1,n_p),C(n_a,n_p-1)\}$$

## 3. Experiments

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/TempCLR3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/TempCLR4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/TempCLR5.png)
