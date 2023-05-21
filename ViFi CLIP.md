# Fine-tuned CLIP Models are Efficient Video Learners

> Rasheed, HanoonaAbdul, et al. “Fine-Tuned CLIP Models Are Efficient Video Learners.” Cornell University - arXiv, Dec. 2022.

## 1 Motivation & Contribution

- 之前的将语言图像模型（CLIP）迁移到视频领域的工作大多采用了Adapter、textual or visual prompts 和跨帧的信息交互模块等等，但这些针对视频特点的设计是否能继续保持原始CLIP模型的泛化性，仍待研究。
- 对于在视频领域full finetuned CLIP模型的研究较少

## 2 Method

### 2.1 Bridge and Prompt

- 本文提出先在K400数据集上full finetuned CLIP模型，通过temporal pooling 的方式融合帧间特征，从而弥补图像和视频间的domain gap，同时保留CLIP模型的泛化能力，即在视频和图像两个不同领域间搭建一个桥梁。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip1.png)

- 为为了测试各种方法对新类别的泛化能力，作者引入了一个基于基类到新类别（Base-to-novel generalization）的视频动作识别泛化设置

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip2.png)

- 实验表明这种full finetuned CLIP 的泛化能力确实比采用其他方法的模型更强。
- 接着在一些小数据集上由于数据量不够，所以采用基于prompt的 frozen CLIP微调范式。

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip3.png)

## 3 Experiment

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip6.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip7.png)

![8](https://raw.githubusercontent.com/bobochow/blog_img/main/img/vificlip8.png)
