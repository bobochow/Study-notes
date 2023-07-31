# The effectiveness of MAE pre-pretraining for billion-scale pretraining

> Singh, Mannat, et al. "The effectiveness of MAE pre-pretraining for billion-scale pretraining." arXiv preprint arXiv:2303.13496 (2023).

## 1 Motivation

- 当前pretrain then finetune的训练范式已经表现出十分不错的效果，其中pretrain阶段主要有两种做法：一种是自监督预训练，另一种则是通常带有噪声数据的弱监督预训练。一些工作尝试将二者结合起来，但训练时依然是独立进行预训练然后微调的两阶段方法。
- 本文尝试通过pre-pretraining的方式将二者结合，简而言之，就是先进行自监督预训练，然后将自监督预训练模型作为弱监督预训练模型的初始化。
- 采用这种方法既能提高不同下游任务的性能，加快模型收敛，又能节省训练时间。
- 本文表明了当今时代模型初始化十分重要。

## 2 MAE -> WSP

- 首先只使用图像进行常规的MAE预训练，然后将互联网图像对应的hash tag作为其弱监督信号，采用多标签分类损失训练模型。

## 3 Experiment

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MAE-WSP1.png)

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MAE-WSP2.png)
