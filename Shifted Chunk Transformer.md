# Shifted Chunk Transformer for Spatio-Temporal Representational Learning

> Zha, Xuefan, et al. "Shifted chunk transformer for spatio-temporal representational learning." Advances in Neural Information Processing Systems 34 (2021): 11384-11396.

## 1 Motivation & Contribution

- 主要目的是为了节约显存，避免整体大量的 Self-Attention。因此提出首先将每帧图片切成大块再变成 patch(类似swin中的window)，并采用哈希注意力减少了计算量，其次在每帧图像自身学完以后通过时序shift操作，联系临近帧进行关联学习。

## 2 Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/shiftchunk1.png)

### 2.1 Image Chunk Self-Attention

- 首先是 Chunk 操作，实际上就是将图片分成几个大块，之后每个大块中划分更小的 patch 做 Self-Attention。做法没有什么变化，先是映射和位置插入，之后经过传统 Encoder 模块，然后将所有的 chunk 拼在一起。
- 将其拼在一起之后，要经过一个全图的 Self-Attention，为了节约显存，这里用的哈希 Attention。
- $\operatorname { softmax }  ( \frac { Q K ^ { T } } { \sqrt { d } _ { k } } )$后，只有少数相近的q,k才有作用，其他的几乎可以忽略不计。
    > [知乎：【哈希 Attention】Reformer: The Efficient Transformer](https://zhuanlan.zhihu.com/p/432341681)
    ---
    > [Reformer 详解](https://www.zhihu.com/tardis/zm/art/105123890?source_id=1005)
    ---

- 之后，为了进一步缩减序列长度，通过一个线性池化层通道数量变成四倍，序列长度变为四分之一。

### 2.2 Shifted Multi-Head Self-Attention

- 考虑到视频中存在的移动，选择使用滑动 Attention，即每一帧的 Attention 不但选择与自己做相关，还会考虑前一帧。

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/shifychunk2.png)

### 2.3 Clip Encoder for Global Clip Attention

- 最后，通过transform对所有的帧级特征做时序特征建模。

## 3 Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/shiftchunk3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/shiftchunk4.png)
