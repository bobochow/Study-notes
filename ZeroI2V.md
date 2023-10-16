# ZeroI2V: Zero-Cost Adaptation of Pre-Trained Transformers from Image to Video

> Li, Xinhao, and Limin Wang. "ZEROI2V: ZERO-COST ADAPTATION OF PRE-TRAINED TRANSFORMERS FROM IMAGE TO VIDEO."

## 1. Motivation

- 受TPS移位思想启发，利用冻结的图像模型同时处理当前帧和其他时刻帧，从而实现图像模型到视频模型的迁移。
- Adapter形式的微调由于其非线性的特点，会提高模型推理的延迟，而LoRA等采用重参数思想的方法则可以避免这一缺点。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ZeroI2V1.png)

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ZeroI2V2.png)

### 2.1 Spatial-temporal Dual-headed Attention

- 受TPS启发，提出了一种多头移位自注意力机制，将当前帧和其他时刻帧的特征进行融合，从而实现图像模型到视频模型的迁移。
- 与TPS的区别是TPS是不同时刻的patch特征通道移位，而本文是自注意力头移位，每个移位头的特征属于单独一帧。
- 而SimDA中的Temporal Patch Shift Attention则将移位patch与当前帧的patch进行拼接后做交叉自注意力，这会导致计算量变大。

### 2.2 Linear Adapter

- Layer merging via structural reparameterization
- 受LoRA启发，将非线性的Adapter层替换为线性的结构重参数化层，从而避免了Adapter层的推理延迟。

- **LoRA**

$$
W_{\text {new }}=W_{\text {LoRA }}+W_{\text {old }}=W_{\text {up }} W_{\text {down }}+W_{\text {old }}
$$

- **Linear Adapter**

$$
W_{\text {new }}=W_{\text {Adapter }} W_{\text {old }}=\left(I+W_{\text {up }} W_{\text {down }}\right) W_{\text {old }}
$$

## 3. Experiments

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ZeroI2V3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ZeroI2V4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ZeroI2V5.png)
