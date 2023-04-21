# MViTv2: Improved Multiscale Vision Transformers for Classification and Detection

> Li, Yanghao, et al. “Improved Multiscale Vision Transformers for Classification and Detection.” arXiv: Computer Vision and Pattern Recognition, Dec. 2021.

---
> <https://zhuanlan.zhihu.com/p/449990416>

## 1 Motication & Contribution

- MViT采用的是和ViT一样的绝对位置编码，这其实忽略了一个很重要的视觉先验知识，即平移不变性(图片中的一个物体无论在图像中如何移动都不会改变这个物体的类别)，绝对位置编码的问题就是物体在图片中移动之后其绝对位置发生了改变，响应的绝对位置编码也发生了变化，所以忽视了平移不变性原理。
- 在MViT中，query pooling只在每个stage的第一个Transformer block中计算，且步长只有(1,2,2)，而Key,Value-pooling步长(1,8,8)在stage所有的Transformer block中计算，相差很多，所以可能会存在Query的信息不足的问题。
- 作者将优化后的MViTv2作为视觉通用模型，应用于图像识别、目标检测、视频识别等任务。

## 2 Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTV2_1.png)

### 2.1 Improved Pooling Attention

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTV2_2.png)

#### 2.1.1 Decomposed relative position embedding

- 绝对位置编码是在进行self-attention计算之前为每一个token添加一个可学习的参数，相对位置编码是在进行self-attention计算时，在计算过程中添加一个可学习的相对位置参数。
- 绝对位置编码缺乏平移不变性，因此作者将相对位置嵌入纳入池化自注意计算中，它只取决于token之间的相对位置距离。
- swin transformer中相对位置编码是每个空间位置都有一个D维的位置向量，计算复杂度=O(H*W)，在改进MViT中，只存储水平及垂直方向的位置向量，其余位置由其所在位置与X,Y轴的相对位置偏差决定，相对位置计算复杂度降低至O(H+W)。

$$\operatorname{Attn} ( Q , K , V ) = \operatorname { S o f t m a x } \Big( ( Q K ^ { \top } + E ^ { ( r e l ) } ) / \sqrt { d } \Big) V$$

$$E _ { i j } ^ { ( r e l ) } = Q _ { i } \cdot R _ { p ( i )  , p ( j )}$$

$$R _ { p ( i ) , p ( j ) } = R _ { h ( i ) , h ( j ) }^{h} + R _ { w ( i ),w(j) } ^ { w }  + R _ { t ( i ) , t ( j ) }^{t}$$

#### 2.1.2 Residual pooling connection

- 为了解决Query pooling与K-V pooling差别过大的问题，采用了Residual pooling connection方式,增加信息流并促进池化注意力的训练。

$$Z : = \operatorname{Attn} ( Q , K , V ) + Q$$

### 2.2 MViT for Object Detection

#### 2.2.1 FPN integration

- MViT的层次结构分四个阶段生成多尺度特征图，因此能很自然地集成到特征金字塔网络(feature Pyramid Networks, FPN)中，用于目标检测任务。

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTv2_3.png)

#### 2.2.2 Hybrid window attention

- MViT通过pooling操作实现了多尺度分层建模并减低了计算量，Swin通过Windows操作同样完成了多尺度建模与计算量减低。
- Swin中采用Windows将patch进行划分，并且Self-attention只在同一窗口中进行计算，大大降低了计算量，为了建立不同窗口之间的关系，采用了shifted windows计算不同窗口之间的Self-attention。
- Hwin也是基于Windows操作的，不过不采用shifted windows操作来建立窗口间的联系，而是换了一种更简单的方式：在同一stage中除了最后一个Transformer blocks之外所有的blocks只进行窗口内Self-attention计算，在最后一个Transformer block中去掉window操作，计算最原始的self-attention，得到混合信息作为输出。

## 3 Experiment

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTv2_4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTv2_5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTv2_6.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTv2_7.png)

![8](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViTv2_8.png)
