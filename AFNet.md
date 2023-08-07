# Look More but Care Less in Video Recognition

> Zhang, Yitian, et al. "Look more but care less in video recognition." Advances in Neural Information Processing Systems 35 (2022): 30813-30825.

## 1 Motivation

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/AFNet1.png)

- 现有的视频识别方法都需要进行帧采样以减少计算量，但是帧采样会导致信息丢失。即使一些方法提出了多种动态显著帧采样策略，但决策过程中被抛弃帧的信息丢失仍然无可避免。此外，动态采样需要额外的决策网络，会增加计算量且使训练过程更加复杂。
- 本文采取了一个和 SlowFast 网络类似的两个分支的网络结构，其中一个ample branch 将所有帧压缩池化得到丰富的信息，另一个focal branch 则选择少量显著帧进行特征提取。两个分支的特征融合后再进行分类。本文的方法不需要额外的决策网络，输入更多帧的同时减少了计算量，且减少了采样带来的信息丢失。
- 与SlowFast的不同：
    1. SlowFast是3D网络，本文是2D网络
    2. 两种方法两个分支的目的不同，SlowFast的两个分支是为了提取不同运动速度的特征，AFNet的两个分支是为了动态选择显著帧，减少计算量。
    3. AFNet为了减少计算量，通过池化降低了空间分辨率，而SlowFast则是减少时间分辨率，空间分辨率不变。
    4. SlowFast通过3D卷积显式地时序建模，而AFNet则是通过特殊设计的模块隐式地时序建模。

## 2 Ample and Focal Network

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/AFNet2.png)

### 2.1 Architecture Design

#### 2.1.1 Ample Branch

- Ample branch 用于以低计算代价提取所有帧中丰富的信息，起到两个作用：
    1. 为 Focal branch 提供选择显著帧的指引信息
    2. 与 Focal branch 起到互补作用，减少信息丢失
- 为了减少计算量，Ample branch 采用了时空2倍下采样的策略。

#### 2.1.2 Navigation Module

- Navigation module 负责显著帧的采样，其根据 上一阶段 Ample branch 的输出，输出一个二值时序掩码，决定哪些帧是输入Focal Branch 的显著帧。
- 首先，采用平均池化和卷积操作将输入特征转换为$T \times 2 \times 1 \times 1$尺寸，然后又将维度重塑为$1 \times （2 \times T） \times 1 \times 1$，最后通过卷积得到一个二值的logit。
$$\tilde{v} _{ y^{a}_ { n } } = \operatorname { R e L U } ( \operatorname { B N } ( W _{ 1 } * \operatorname{Pool} ( v_ { y^{a} _ { n } } ) ) )$$

$$p _ { n } ^ { t } = W _ { 2 } * \tilde{v} _ { y^{a} _ { n } }$$

- 然而，直接从这样的离散分布中采样是不可微的，本文通过引入Gumbel-Softmax来解决这个问题。
- 首先通过Softmax生成一个归一化的类别分布
$$\pi = \{ l _ { j } | l _ { j } = \frac { \exp ( p _ { n } ^ { t _ { j } } ) } { \exp ( p _ { n } ^ { t _ { 0 } } ) + \exp ( p _ { n } ^ { t _ { 1 } } ) } \}$$
-然后从Gumbel 分布中采样并通过argmax得到离散值

$$L =  \underset{j}{\operatorname { a r g m a x}} ( \log l _ { j } + G _ { j } )$$

$$G _ { j } = - \log ( - \log U _ { j } )$$

- 由于arg max不可导，因此在反向传播过程中使用SoftMax代替arg max

$$\hat{l} _ { j } = \frac { \exp ( ( \log l _ { j } + G _ { j } ) / \tau ) } { \sum _ { k = 1 } ^ { 2 } \exp ( ( \log l _ { k } + G _ { k } ) / \tau ) }$$

> 知乎：<https://zhuanlan.zhihu.com/p/59550457>

#### 2.1.3 Focal Branch

- 在 Navigation Module 的引导下，Focal Branch只计算选定的帧，从而减少了计算成本和冗余帧的潜在噪声。
- 最后通过动态权重将两个分支融合。

### 2.2 Spatial Redundancy Reduction

- 这种Gumbel-Softmax采样策略也可以用于空间维度的采样，从而减少空间冗余。

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/AFNet3.png)

## 3 Experiment

- Less is more
![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/AFNet5.png)

- More is less
![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/AFNet4.png)
