# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> Liu, Ze, et al. “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows.” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2022, <https://doi.org/10.1109/iccv48922.2021.00986>.

---
> [知乎](https://zhuanlan.zhihu.com/p/430047908)

## 1 Motivation & Contribution

- 为了解决一般ViT无法有效处理多分辨率图像且计算量很大的问题，swin transformer基于ViT模型，引入了CNN中滑动窗口机制和多尺度设计的思想，并且大大减少了transformer的计算量。

## 2 Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/Swin3.png)

### 2.1 Overall Architecture

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin5.png)

- 整个模型采取层次化的设计，一共包含4个Stage，每个stage都会通过 patch merging 缩小输入特征图的分辨率，像CNN一样逐层扩大通道数。

- patch merging：每次H/W降采样是两倍，通道维度则会变成原先的4倍，再通过一个全连接层再调整通道维度为原来的两倍。

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin7.png)

### 2.2 Shifted Window based Self-Attention

- 传统的Transformer都是基于全局来计算注意力的，因此计算复杂度十分高。而Swin Transformer则将注意力的计算限制在每个窗口内，进而减少了计算量，引入了window-multihead self attention。但不重叠的局部窗口缺乏全局联系，因此通过引入shifted window multihead self-attention，既引入了跨窗口联系，又保持了窗口注意力的低计算量。

#### 2.2.1 Self-attention in non-overlapped windows

##### 2.2.1.1 Computational Complexity

- window attention使得计算复杂度由patches数量的平方关系降低到线性关系。

$$\Omega ( \operatorname{MSA} ) = 4 h w C ^ { 2 } + 2 ( h w ) ^ { 2 } C$$

$$\Omega ( \operatorname{W-MSA} ) = 4 h w C ^ { 2 } + 2 M^2 h w  C$$

##### 2.2.1.2 Window Partition/Reverse

- window partition函数是用于对张量划分窗口，指定窗口大小。window reverse函数则是对应的逆过程。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin11.png)

##### 2.2.1.3 Relative position bias

- Swin transformer没有采用标准transformer中的绝对位置编码，而是采用了相对位置编码。

$$\operatorname{Attention}( Q , K , V ) = \operatorname { S o f t M a x } ( Q K ^ { \top } / \sqrt { d } + B ) V$$

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/Swin1.png)

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/Swin2.png)

#### 2.2.2 Shifted window partitioning in successive blocks

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/Swin4.png)

- 以$(M/2,M/2)$向下取整的窗口重新对原图进行分割，并将之前没有联系的新窗口合并得到新的窗口划分方案。由此得到了两个连续的swin transformer blocks，前一层做W-MSA，后一层做SW-MSA。

$$\hat{\mathbf{z}}^ { l } =\operatorname{W - M S A} ( \operatorname{LN}L N ( \mathbf{z} ^ { l - 1 } ) ) + \mathbf{z}^ { l - 1 }$$

$${\mathbf{z}}^ { l } =\operatorname{MLP} ( \operatorname{LN}L N ( \mathbf{\mathbf{\hat{z}}} ^ { l } ) ) + \mathbf{\hat{z}}^ { l }$$

$$\hat{\mathbf{z}}^ { l + 1 } =\operatorname{SW - M S A} ( \operatorname{LN}L N ( \mathbf{z} ^ { l } ) ) + \mathbf{z}^ { l }$$

$${\mathbf{z}}^ { l + 1 } =\operatorname{MLP} ( \operatorname{LN}L N ( \mathbf{\mathbf{\hat{z}}} ^ { l + 1 } ) ) + \mathbf{\hat{z}}^ { l + 1 }$$

##### 2.2.2.1 Efficient batch computation for shifted configuration

- 随之而来的问题就是窗口个数增加了，为了避免窗口增加导致的额外计算量并保证不重叠窗口间有关联，论文提出了cyclic shift方法。
- cyclic shift：通过对特征图移位，并给 Attention 设置 mask 来间接实现的。能在保持原有的 window 个数下，最后的计算结果等价。

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin6.png)

- torch.roll：

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin8.png)

- mask attention：在新的窗口中，只有相同窗口的部分才能计算self-attention，不同窗口间计算的self-attention需要归0，根据self-attention公式，最后需要进行Softmax操作，不同窗口间计算的self-attention结果通过mask加上-100，在Softmax计算过程中，Softmax(-100)无线趋近于0，达到归0的效果。

![10](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin10.png)

```python
tensor([[[[[   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.],
           [   0.,    0.,    0.,    0.]]],


         [[[   0., -100.,    0., -100.],
           [-100.,    0., -100.,    0.],
           [   0., -100.,    0., -100.],
           [-100.,    0., -100.,    0.]]],


         [[[   0.,    0., -100., -100.],
           [   0.,    0., -100., -100.],
           [-100., -100.,    0.,    0.],
           [-100., -100.,    0.,    0.]]],


         [[[   0., -100., -100., -100.],
           [-100.,    0., -100., -100.],
           [-100., -100.,    0., -100.],
           [-100., -100., -100.,    0.]]]]])
```

### 2.3 Flow chart

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin12.png)

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin13.png)

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin15.png)

## 3 Experiment

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swin16.png)

---

## **Video Swin Transformer**

> Liu, Ze, et al. “Video Swin Transformer.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, <https://doi.org/10.1109/cvpr52688.2022.00320>.

### 1 Architecture

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/videoswin1.png)

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/videoswin2.png)

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/videoswin3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/videoswin4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/videoswin5.png)

## Swin Transformer V2: Scaling Up Capacity and Resolution

> Liu, Ze, et al. “Swin Transformer V2: Scaling Up Capacity and Resolution.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, <https://doi.org/10.1109/cvpr52688.2022.01170>.

### 1 Motivation

- SwinTransformer-v1在视觉领域里并没有像NLP那样，对于增大模型scale有比较好的探索。原因主要是：
  - 在增大视觉模型的同时可能会带来很大的训练不稳定性
  - 在很多需要高分辨率的下游任务上，还没有很好的探索出来对低分辨率下训练好的模型迁移到更大scale模型上的方法
  - GPU memory cost太大

### 2 Swin Transformer V2

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swinv2_1.png)

#### 2.1 Scaling Up Model Capacity

##### 2.1.1 Post normalization

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swinv2_2.png)

- 当我们扩大模型容量时，在更深层观察到激活值的显著增加。实际上，在预归一化配置中，每个残差块的输出激活值直接合并回主支路，并且主支路的振幅在更深的层上变得越来越大。各层振幅差异大，导致训练失稳。因此，将pre norm 改为了post norm。

##### 2.1.2 Scaled cosine attention

- 对于q / k 向量相乘后，因为要经过一个 Softmax 函数，实际上大部分值小的向量就会变为 0 或者很小值，在最后根本不会起作用。。因此，作者将点积自注意力改为了余弦自注意力。

$$\operatorname{Sim} ( q _ { i } , k _ { j } ) = \cos ( q _ { i } , k _ { j } ) / \tau + B _ { i j }$$

#### 2.2 Scaling Up Window Resolution

##### 2.2.1 Continuous relative position bias

- 模型使用更大的图像与更大的窗口，训练后的精度就会下降的很严重。原因是将低分辨率的相对位置编码通过双三次插值迁移到大分辨率窗口时，输进编码函数的相对位置过大了。
- 因此，提出了连续型的位置编码。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swinv2_3.png)

##### 2.2.2 Log-spaced coordinates

- 当在很大程度上不同的窗口大小之间传递时，将会有很大一部分的相对坐标范围需要插值操作。因此，使用对数空间坐标来代替原来的线性空间坐标来解决这个问题：

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swinv2_4.png)

### 3 Experimet

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/swinv2_5.png)
