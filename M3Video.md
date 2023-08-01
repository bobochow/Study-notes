# M^3 Video: Masked Motion Modeling for Self-Supervised Video Representation Learning

> Sun, Xinyu, et al. "M $^ 3$ Video: Masked Motion Modeling for Self-Supervised Video Representation Learning." arXiv preprint arXiv:2210.06096 (2022).

## 1 Motivation

- 当前的一些掩码视频模型重建目标依然局限于图像空间信息（如原始RGB像素、手工特征和离散VQ-VAE嵌入），这存在着忽视时序信息以及运动细节的问题。
- 本文认为通过引入预测运动轨迹（包含位置偏移和形状变化）可以弥补运动信息的缺失，掩码模型重建目标不再是低层次的图像特征，而是高层次的运动轨迹特征。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/M3video1.png)

## 2 Masked Motion Modeling

- MAE与传统IDT的结合

### 2.1 General Scheme of $\text{M}^3\text{Video}$

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/M3video2.png)

- 模型前半部分与VideoMAE保持一致，采用管道式掩码策略，encoder只输入非掩码patches，encoder输出特征与可学习的mask tokens被输入decoder。
- 不同之处在于$M^3Video$重建目标是masked patches 中根据采样的运动轨迹提取的运动特征。
- 因此，训练损失函数为：

$$\mathcal{L} = \sum _{ i \in \mathcal{I} } | z_ { i } - \hat{z} _ { i } | ^ { 2 } \\
\text{where $\hat{z}$ is the predicted motion target, and $\mathcal{I}$ is the index set of motion targets in all masked patches.}
$$

### 2.2 Trajectory Motion Target for Mask Modeling

- 传统手工特征（HOF,MBH）和光流都只能描述短期两帧间的特征，因此为了捕捉长期特征，本文采用与DT算法一致的方法，即在L帧中采样跟踪点轨迹并沿轨迹提取特征。
- 对于采样帧间隔大于1帧的轨迹，本文借鉴视频插帧的思想进行插值细化，使得模型能学习到更细致的运动信息。

#### 2.2.1 Tracking objects using spatially and temporally dense trajectories

- 本文一个大小为$t\times h \times w$的masked patch中均匀采样K个点，依据密集光流跟踪得到L帧中的K个轨迹。

$$T = ( p _ { t } , p _ { t + 1 } , \cdots , p _ { t + L } )$$

- 依据轨迹提取位置特征$z^p$和形状特征$z^s$

$$z = ( z ^ { p } , z ^ { s } )$$

#### 2.2.2 Representing Position Features

- 轨迹特征由归一化的相对位移组成

$$
\Delta p _ { t } = p _ { t + 1 } - p _ { t }
\\
z ^ { p } = ( \Delta p t , \cdots , \Delta p _ { t + L - 1 } )$$

#### 2.2.3 Representing Shape Features

- 本文与DT算法一样提取与轨迹对齐的HOG特征

$$z ^ { s } = ( H O G ( p _ { t } ) , \cdots , H O G  ( p _ { t + L - 1 } ) $$

## 3 Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/M3video3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/M3video4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/M3video5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/M3video6.png)
