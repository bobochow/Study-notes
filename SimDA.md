# SimDA: Simple Diffusion Adapter for Efficient Video Generation

> Xing, Zhen, et al. "SimDA: Simple Diffusion Adapter for Efficient Video Generation." arXiv preprint arXiv:2308.09710 (2023).

## 1. Introduction

- 通过引入了Temporal Patch Shift和文本信息来微调图像生成模型，以提高视频生成质量和效率。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/SimDA1.png)

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/SimDA2.png)

- 其中的Temporal Patch Shift Attention 与TPS的区别除了window attention 与cross attention 外还未知。

## 3. Experiments

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/SimDA3.png)
