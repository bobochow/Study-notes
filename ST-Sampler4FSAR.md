# Task-adaptive Spatial-Temporal Video Sampler for Few-shot Action Recognition

> Liu, Huabin, et al. "Task-adaptive Spatial-Temporal Video Sampler for Few-shot Action Recognition." Proceedings of the 30th ACM International Conference on Multimedia. 2022.

## 1. Motivation & Contribution

- 当前的主流少样本动作识别方法主要集中于增强视频特征以及改进对齐度量，对于模型输入数据的关注较少。当前的帧采样策略往往忽视了视频中重要的时空信息，无法有效地利用有限的数据。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ST-Sampler4fsar1.png)

- 本文引入了一种视频帧采样方法，以增强视频帧的利用。通过时间选择器（temporal selector）和空间放大器（spatial amplifier）的组合，识别关键帧并强调其显著性信息。该方法还融入了任务自适应学习，根据特定任务动态调整帧采样策略。

## 2. Method

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ST-Sampler4fsar2.png)

### 2.1 Video scanning

- 为了充分发掘整个视频中的时空信息，本文首先使用一个轻量级的ShuffleNet-v2提取密集采样的低分辨率视频帧的特征$f_X$，而视频级的特征表示$g_X$则为所有帧特征的平均值。

### 2.2 Temporal selector

#### 2.2.1 Evaluation

- 为了从密集采样的M帧中选择出T帧关键帧，TS模块首先通过一个evaluator $\Phi$来评估每一帧的重要性。

$$\begin{aligned}s_i&=\Phi(\operatorname{Avg}(\operatorname{Cat}(f(X_i),g_\mathbf{X}))+\operatorname{PE}(i))\\\Phi&=w_2(\operatorname{ReLU}(w_1(\cdot)))\end{aligned}$$

- 其中$w_2$会随着episode动态调整
- 接着，根据重要性分数，本文采用了一个可微分的Top-K选择器来选择出T帧关键帧。
- 将被选择帧的序号从小到大排列，并转换为独热编码$\mathbf{I}=\{I_{i_{1}},I_{i_{2}},\ldots,I_{i_{T}}\}\in\{0,1\}^{M\times T}$，由此可以通过矩阵乘法得到被选择的子集

$$\mathbf{X}^{\prime}=\mathbf{I}^T\mathbf{X}$$

#### 2.2.2 Differentiable selection

- 由于Top-K选择和独热编码是不可微分的，因此本文采用了一种最大摄动方法使得训练过程中采样过程可微分。
- 最大摄动是Gumbel-max的推广，与[AFNet](https://arxiv.org/abs/2211.09992)
类似。
- 时序选择与求解下列线性规划等价：

$$\underset{\mathrm{I}\in\mathcal{C}}{\operatorname*{\arg\max}}\langle\mathrm{I},\mathrm{S}1^T\rangle $$

- **Forward**:

$$\mathbf{I}_\sigma=\mathbb{E}_Z\left[\underset{\mathrm{I}\in\mathcal{C}}{\operatorname*{\arg\max}}\left\langle\mathbf{I},\mathbf{S1}^\mathrm{T}+\sigma\mathbf{Z}\right\rangle\right]$$

- **Backward**:

$$J_\mathbf{s}\mathbf{I}=\mathbb{E}_Z\left[\underset{\mathrm{I}\in\mathcal{C}}{\operatorname*{\arg\max}}\left\langle\mathbf{I},\mathbf{S1}^\mathrm{T}+\sigma\mathbf{Z}\right\rangle\mathbf{Z}^\mathrm{T}/\sigma\right]$$

- 推理时直接采用hard Top-K选择减少计算量，但这与训练过程不一致，因此训练时逐步衰减$\sigma$至0。

### 2.3 Spatial amplifier

- 参考[知乎2019CVPR_Trilinear Attention Sampling Network解读](https://zhuanlan.zhihu.com/p/62737422)

- 大多数主流方法对于视频空间不同区域是一视同仁的，但视频存在空间冗余性，模型应更关注视频中的关键区域。因此本文采用了一种基于注意力的非均匀抽样。

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ST-Sampler4fsar3.png)

#### 2.3.1 Saliency map generation

- 首先根据自注意力机制汇聚所有通道特征图，生成空间显著性图

$$\alpha=\frac{f(X)f(X)^T}{\sqrt{H\times W}}\in\mathbb{R}^{C\times C},f(X)^{\prime}=\alpha f(X)\in\mathbb{R}^{C\times H\times W}$$

$$\mathbf{M_s}\in\mathbb{R}^{H\times W}=\frac1C\sum_i^Cw_{s_i}\cdot f(X)^{\prime}$$

#### 2.3.2 Amplification by 2-D inverse sampling

- 本文通过累积分布函数的反函数进行非均匀采样，以放大显著区域的特征。
- 首先将显著图按坐标轴最大值分解(a1),(a2)

$$M_x=\max_{1\leq i\leq W}(\mathbf{M_s})_{i,j},M_y=\max_{1\leq j\leq H}(\mathbf{M_s})_{i,j}$$

- 然后通过Cumulative Distribution Function (cdf)获得坐标轴相对分布(b1),(b2)

$$D_x=\operatorname{cdf}(M_x),D_y=\operatorname{cdf}(M_y)$$

- 最后根据cdf反函数获得非均匀采样点，再根据仿射变换得到最终的放大图像

$$X_{i,j}^{'}=\operatorname{Func}(F,\mathbf{M_s},i,j)=X_{D_x^{-1}(i),D_y^{-1}(j)}$$

### 2.4 Task-adaptive sampling learner

- 一般地用于动作识别的采样策略训练后往往是固定的，但是这样的方式并不适用于少样本场景，因此本文对采样器采用了一个任务自适应学习器，它为 TS 和 SA 中的层生成特定于任务的参数，根据手头的episode task动态调整采样策略。

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ST-Sampler4fsar4.png)

- 首先将其参数化为具有对角协方差的条件多元高斯分布，通过一个有两层线性层的编码器估计该任务的均值和方差。

$$\mu,\sigma=\frac1{N\times K}\sum_{i=1}^{N\times K}E(\mathrm{Avg}(g_{\mathrm{X}^i}))$$

- 接下来通过重参数化技巧$f_t\in\mathbb{R}^{128}=\mu+\sigma\varepsilon$ ，采样得到符合下列任务分布的任务总体特征。

$$p(f_t\mid\mathcal{S})=\mathcal{N}\left(\mu,\operatorname{diag}\left(\sigma^2\right)\right)$$

- 最后，根据任务总体特征生成用于调整采样器的任务特定参数

$$\theta_{ts}=G_t(f_t),\theta_{sa}=G_s(f_t)$$

$$w_{2}\leftarrow\frac{\theta_{ts}}{\|\theta_{ts}\|},w_{s}\leftarrow\frac{\theta_{sa}}{\|\theta_{sa}\|}$$

### 2.5 Optimization

- 本文基于ProtoNet实现，query特征和prototype特征则与scan-Net生成的全局特征通过Cat结合，增强特征表示。
- 损失函数除了常规的交叉熵损失外，还增加了一个额外的分类辅助损失。

## 3. Experiments

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ST-Sampler4fsar5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ST-Sampler4fsar6.png)
