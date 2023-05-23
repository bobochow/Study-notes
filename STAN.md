# Revisiting Temporal Modeling for CLIP-based Image-to-Video Knowledge Transferring

> Liu, Ruyang, et al. Revisiting Temporal Modeling for CLIP-Based Image-to-Video Knowledge Transferring. Jan. 2023.

## 1. Motivation

- 将CLIP模型迁移到视频领域的关键是利用预训练的CLIP模型知识建模时序信息。
- 视频领域里的不同下游任务对此有不同的做法。视频文本检索通常将CLIP作为特征提取器，对提取的帧级嵌入做时序建模，得到较高层次的语义嵌入。但其无法有效捕捉低层次的时空特征变化。
- 而视频识别任务大多在CLIP模型中嵌入建模时序特征的模块，从而增强了CLIP模型建模时空特征的能力。但嵌入额外的时序建模模块会破坏CLIP模型自身的高层次语义知识。
- 这两个不同领域的模型通常无法在两个领域同时取得较好的效果，但显然对于视频理解来说，低层次和高层次语义信息都是需要的，如何构建一个能同时获取两种层次特征的模型需要进一步的研究。
- 本文受FPN模型的启发，通过增加一个旁支网络，避免影响原本网络模型的性能。此外通过将原始网络不同层次的特征输入旁支网络，也能建模不同层次的语义特征。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STAN1.png)

## 2. Method

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STAN2.png)

- 模型方法与EVL类似，将CLIP模型作为特征提取器，将CLIP提取的不同层次的特征输入旁支网络，通过3D Conv 和时序自注意力建模时序特征，不同的是复用了CLIP模型中的image encoder部分，增强了时空特征。

## 3. Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STAN3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/STAN4.png)
