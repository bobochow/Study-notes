# Can an image classifier suffice for action recognition?

> Fan, Quanfu, Chun-Fu Chen, and Rameswar Panda. "Can an image classifier suffice for action recognition?." International Conference on Learning Representations. 2022.

## 1. Motivation

- 尝试研究只使用图像分类模型（不增加额外的时序建模模块）能否胜任视频分类任务。
- 通过采用类似DualPath中 temporal path 的 grid-like frame set（此文中称为super image）的方式，使用图像模型构建时序关系。
- 如何有效学习grid-liek frame set中的时空学习仍待深入研究，因为在这一图像模式中既涉及到单张图像窗口中的局部空间信息，又包含了跨帧的全局信息。这是相对复杂，作者选择了较为符合上述要求的swin fransformer作为backbone。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/sifar.png)

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/sifar1.png)

### 2.1 Sliding Window

- 如Figure 3所示，SIFAR沿用Swin transformer的siding window，以实现帧内和跨帧特征提取。swin transformer 每个stage自带的patch merging(pooling)则会进一步扩大感受野。
- 对除最后一层之外的所有层保持与swin transformer 相同的窗口大小，最后一层窗口与其图像分辨率一样大，做全局注意力，提高所有帧间的信息融合。

### 2.2 Creation of Super Image

- 如Figure 4所示，SIFAR将按顺序排列的n张图像组合为一张超大分辨率的图像。
- 实验结果表明，更紧凑的结构，如正方形网格，可以促进模型学习跨帧的时间依赖性，因为这样的形状提供了任意两个图像之间的最短最大距离。

## 3. Difference between SIFAR and Dual path

- SIFAR只使用图像分类模型，没有使用其他复杂的时间建模模块。而Dual Path则是一个双流模型，且在CLIP模型基础上添加了时空Adapter。
- SIFAR的backbone是IN-21K预训练的 Swin transformer，Dual path则为多模态预训练大模型CLIP，含有更丰富的语义信息。
- SIFAR的super image分辨率很大，Dual Path的grid-like frame set resize到原图大小。

## 4. Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/sifar2.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/sifar3.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/sifar4.png)
