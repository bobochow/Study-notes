# Frame Flexible Network

> Zhang, Yitian, et al. Frame Flexible Network. Mar. 2023.
> >[(知乎) CVPR-2023 | FFN: 针对视频识别的通用Once-For-All框架](https://zhuanlan.zhihu.com/p/624090867)

## 1 Motivation & Contribution

- **动机**：视频识别通常会采样多帧图像来代表整个视频。现有的视频识别算法总是对具有不同帧数的输入分别进行训练，这需要重复的训练操作和成倍的存储成本。

- **观察**：如果我们在模型推理的时候使用训练未用到的帧数，模型性能则会显著下降（见下图），这被总结为时域频率偏移现象。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/FFN1.png)

- **解决方案**：提出了一个通用的框架，名为Frame Flexible Network（FFN），它不仅可以使模型根据输入帧数的不同从而动态地调节计算量，还可以显著减少存储多个模型的内存成本。
- **优点**：（1）一次性训练（2）明显的性能增益（3）参数量的显著节省（4）计算量的动态调整（5）强大的兼容性。

## 2 Temporal Frequency Deviation

- 现有视频识别方法将相同的网络对不同帧的输入分别进行训练，以获得具有不同性能和计算量的多个模型。但将模型在高帧数进行训练，然后直接在较少帧数的输入上进行推理，模型效果会变差。作者将这种普遍存在的现象称为时域频率偏移。

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/FFN2.png)

### 2.1 Nearby Alleviation

- 实验发现，如果推理帧数接近于训练帧数，性能差距会更小，作者将这个现象称之为相邻缓解（Nearby Alleviation）。

### 2.2 Normalization Shifting

- 不同帧输入所对应的feature map的统计值存在一定的差异，也就意味着模型在推理的时候使用其他帧数的输入会导致Normalization Shifting，这是作者认为造成时域频率偏移现象的主要原因。

## 3 Method

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/FFN3.png)

- 为了让网络能够在任意帧数推理的时候表现出与Separated Training（ST）相似或者更好的性能，所以作者在训练的时候引入多个视频序列（对同一个视频采样不同帧数得到）并建立对应的子网络，然后在模型推理的时候根据输入帧数的大小来激活对应的子网络，防止Normalization Shifting的产生。

### 3.1 Multi-Frequency Alignment

- Weight Sharing：对于任意视频识别网络，我们可以将其拆分为两部分，一部分是用于空间-时序建模的模块，剩下的部分就是各种Normalization的操作。Normalization Shifting是导致时域频率偏移的主要原因，因此我们可以共享空间-时序建模模块的参数，而对每个子网络分配不同的Normalization操作。
- Temporal Distillation：给定视频$v$
 ，我们对其按帧数大小的不同进行采样，分别得到$v^{L},v^{M}和v^{H}$, 然后三个输入会分别进入对应的子网络，并得到对应的输出$p^{L},p^{M}和p^{H}$。由于$v^{H}$采样的帧数最多，因此$p^{H}$ 也会对应更好的结果，所以我们对$p^{H}$ 计算Cross-Entropy Loss来更新对应的子网络参数，然后分别计算$p^{M}$和$p^{H}$以及$p^{L}$和$p^{H}$的KL Divergence Loss来更新剩下子网络的参数。这样就可以迫使$p^{L},p^{M}和p^{H}$ 三者尽可能的相近，即让不同帧数的输入产生相似的输出，也就是让模型学习与帧数变化无关的表达（Temporal Frequency Invariant Representation）。
- 交叉熵使得模型能尽可能低区分不同类别的视频，而KL散度则让模型识别不同帧率的同一类别的视频。

### 3.2 Multi-Frequency Adaptation

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/FFN4.png)

- **Weight Alteration**：为了进一步增强模型的表达能力，提出了Weight Alteration，一个简单的深度可分离卷积层，并将其插入到各个子网络的不同stage当中，这样我们就可以通过一个简单的线性变换使得各个子网络拥有一套属于自己的独特参数。

### 3.3 Inference at Any Frame

- 给定推理的帧数n，我们会比较n与L, M, H的距离，并选择与其距离最小的子网络进行激活，来得到对应的输出。

## 4 Experiment

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/FFN5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/FFN6.png)
