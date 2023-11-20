# Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning

> Lian, Dongze, et al. "Scaling & shifting your features: A new baseline for efficient model tuning." Advances in Neural Information Processing Systems 35 (2022): 109-123.

## 1. Motivation

- 大多数参数高效微调方法都会改变特定的主干架构或网络的输入，这可能会导致频繁的结构修改和繁重的工作量。
- 本文介绍了一种新的参数高效的微调方法SSF，即只需通过缩放和位移预训练模型提取的深度特征，即可达到全参数微调的性能水平。与其他参数高效的微调方法相比，SSF在可调参数数量更少的情况下也能取得更好的性能。此外，SSF只会在训练阶段添加可学习参数，并且这些额外的参数可以通过重新参数化合并到原始预训练模型权重中进行推理。

## 2. Method

### 2.1 The design of SSF

![1](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/SSF1.png)

![2](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/SSF2.png)

### 2.2 Re-parameterization

- Training:

$$y=\gamma\odot x+\beta $$

- Inference:

$$y=\gamma\odot x+\beta=\gamma\odot(w*t+b)+\beta=(\gamma\odot w)*t+\gamma\odot b+\beta $$

## 3. Experiments

![3](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/SSF3.png)

![4](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/SSF4.png)

![5](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/SSF5.png)
