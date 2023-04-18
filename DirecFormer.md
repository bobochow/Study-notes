# DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition

> Truong, Thanh-Dat, et al. "Direcformer: A directed attention in transformer approach to robust action recognition." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

## 1 Motivation & Contribution

- 在transformer模型中研究了学习视频帧顺序的问题，即给定乱序视频也能纠正顺序并给出准确类别预测。
- 过去的工作主要基于CNN模型采用乱序帧代理任务的自监督学习方法。而本文则基于ViT中的spatio-temporal attentive patch，通过新的自监督损失函数，给注意力提供了方向信息。
- Video Ordering：之前的工作大多采用自监督方式，监督形式较弱，通常是二进制的有序或无序标签，或者是有序的sub-clip。

## 2 Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/DirecFormer.png)

### 2.1 Goal

- 作者希望DirecFormer不仅能分类动作，还能推理经过顺序变化后的视频的正确顺序。

$$
\arg \max _\theta \mathbb{E}_{\mathbf{x}, \mathbf{y}, \mathbf{o}, \mathbf{i}}(\log (p(\mathbf{y} \mid \mathbf{x} ; \theta))+\log (p(\mathbf{i} \mid \mathcal{T}(\mathbf{x}, \mathbf{o}) ; \theta)))
$$

$$\hat{\mathbf{y}} = \phi _ { cls } \odot \mathcal{G}  ( \mathbf{x} )$$

$$\hat{\mathbf{i}}= \phi _{ord}\odot \mathcal{G}(\mathcal{T}(\mathbf{x,o})) $$

### 2.2 Directed Attention Approach

- 原始的transformer中采用的 scaled dot attention 只能简单地表示 token 之间的相关性但忽略了token之间的时间或空间顺序。因此，作者没有采用传统的softmax注意力，而是加入了余弦相似度，使得自注意力带有了方向信息。
- 为了减少计算量，采用了divided spatial temporal attention，先做temporal后做spatial。

- temporal directed attention :

$$\mathbf{a}_{(s, t)}^{(l) - time}=\left[\cos \left(\frac{\mathbf{q}_{s, t}^{(l)}}{\sqrt{D}}, \mathbf{k}_{0,0}^{(l)}\right)\left\{\cos \left(\frac{\mathbf{q}_{s, t}^{(l)}}{\sqrt{D}}, \mathbf{k}_{s, t^{\prime}}^{(l)}\right)\right\}_{t^{\prime}=1}^T\right]$$

$$ \mathbf{s}_{s,t}^{(l)-time}=\mathbf{a}_{(s,t),(0,0)}^{(l)-time}\mathbf{v}_{0,0}^{(l)}+\sum_{t'=1}^{T}\mathbf{a}_{(s,t),(s,t')}^{(l)-time}\mathbf{v}^{(l)}_{s,t'}$$

$$\mathbf{z'}^{(l)-time}_{s,t}=\mathbf{z}_{s,t}^{(l-1)}+\gamma^{(l)-time}(\mathbf{s}_{s,t}^{(l)-time})$$

- spatial directed attention :

$$\mathbf{a}_{(s, t)}^{(l)- space}=\left[\cos \left(\frac{\mathbf{q'}_{s, t}^{(l)}}{\sqrt{D}}, \mathbf{k'}_{0,0}^{(l)}\right)\left\{\cos \left(\frac{\mathbf{q'}_{s, t}^{(l)}}{\sqrt{D}}, \mathbf{k'}_{s, t^{\prime}}^{(l)}\right)\right\}_{t^{\prime}=1}^T\right] $$

$$ \mathbf{s}_{s,t}^{(l)-space}=\mathbf{a}_{(s,t),(0,0)}^{(l)-space}\mathbf{v}_{0,0}^{(l)}+\sum_{t'=1}^{N}\mathbf{a}_{(s,t),(s',t)}^{(l)-time}\mathbf{v}^{(l)}_{s',t}$$

$$\mathbf{z'}^{(l)-space}_{s,t}=\mathbf{z'}_{s,t}^{(l)-time}+\gamma^{(l)-space}(\mathbf{s}_{s,t}^{(l)-space})$$

- MLP :

$$\mathbf{z}^{(l)}_{s,t}=\varphi^{(l)}(\tau^{(l)}(\mathbf{z'}^{(l)-space}_{s,t}))+\mathbf{z'}^{(l)-sapce}_{s,t} $$

- Classification Embedding :

$$\hat{\mathbf{y}} = \phi _ { cls } (\tau_{cls}(\mathbf{z}_{0.0}^{(L)}))$$

$$\hat{\mathbf{i}}= \phi _{ord}\odot \mathcal{G}(\mathcal{T}(\mathbf{x,o})) $$

### 2.3 Self-supervised Guided Loss For Directed Temporal Attention Loss

- 提出了一个自监督损失函数，显式地从先验顺序知识中强制执行时间注意学习

$$\mathcal{L}_{self}=\frac{1}{LNT^2}\sum_{l=1}^{L}\sum^{N,T}_{s=1,t=1}\sum^{T}_{t'=1}(1-\mathbf{a}^{(l)-time}_{(s,t),(s,t')})\varsigma(\mathbf{o}_t,\mathbf{o}_{t'}) $$

$$
\varsigma(\mathbf{o}_t,\mathbf{o}_{t'})=\begin{cases}
   1 &\text{if } { \mathbf{o}_t <\mathbf{o}_{t'} } \\
   -1 &\text{others }
\end{cases}
$$

$$\mathcal{L}=\lambda_{cls}\mathcal{L}_{cls}+\lambda_{ord}\mathcal{L}_{ord}+\lambda_{self}\mathcal{L}_{self} $$

## 3 Experiment

- Frame permutation：从T！=8！中选择了1,000个随机排列用于学习时间顺序变换。
- Order Correction：将输出的时序注意力（即各帧间的余弦相似度）看作图的邻接矩阵，每一帧看作图中的节点。在图中搜索权重最大的哈密尔顿路径，即为更正的顺序

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/DirecFormer_5.png)

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/DirecFormer_2.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/DirecFormer_3.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/DirecFormer_4.png)
