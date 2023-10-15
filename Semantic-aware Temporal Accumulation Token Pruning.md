# Prune Spatio-temporal Tokens by Semantic-aware Temporal Accumulation

> Ding, Shuangrui, et al. "Prune spatio-temporal tokens by semantic-aware temporal accumulation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

## 1. Motivation

- 对于视频存在的时空冗余问题，目前仍没有很好的解决方案。
- 基于Transformer的动作识别方法已取得了令人瞩目的成果，但其巨大的计算量阻碍了其进一步发展。
- 已经有一些方法尝试通过剪枝来减少计算量，但是这些方法多是借鉴了图像领域的方法，是基于空间维度的，往往忽略了时间维度的冗余。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STA-pruning1.png)

- 本文提出了一种语义感知的时序累积重要性分数，用于剪枝视频中的时空token。
- 该分数由两部分组成：时序冗余分数和语义感知分数，目的是丢弃那些时序冗余且空间语义不重要的token。

### 2.1 Temporal redundancy

- 本文将时序冗余定义为相邻帧间的相似tokens，因此通过计算相邻两帧间的token相似度与时序累积分数来衡量时序冗余。

### 2.2 Semantic-awareness

- 本文将语义感知定义为token的空间重要性，因此通过基于激活值的自注意力图，计算token的空间重要性来衡量语义感知。

### 2.3 Algorithm

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STA-pruning2.png)

- 读者按：相似度的计算量又如何减少？

## 3. Experiments

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STA-pruning3.png)
