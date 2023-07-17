# Optimizing ViViT Training: Time and Memory Reduction for Action Recognition

> Gowda, ShreyankN, et al. Optimizing ViViT Training: Time and Memory Reduction for Action Recognition. June 2023.

## 1 Motivation & Contribution

- 通过冻结预训练的spatial transformer部分，然后引入adapter模块再进行微调的方式，减少了ViViT的训练耗时以及开销的同时保持模型精度。
- 与流行的微调CLIP模型训练范式类似，只是模型不同。

## 2 Two Stage Training Strategy

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/SFA-VIVIT1.png)

### 2.1 Stage 1 : Pretaining

- 在第一阶段，使用预训练的图像模型作为spatial transformer的初始化，并采用较少的输入帧数（8帧），得到一个初步平衡了效率和精度的模型。

### 2.2 Stage 2 ：Finetuning

- 在第二阶段，使用第一阶段的模型的spatial ＆ temporal transformer 作为二阶段模型的初始化，并且将spatial transformer 的参数冻结，在其后加入 spatial adapter 模块。此外，二阶段输入更多的帧数（128帧）。整个模型仅需微调adapter模块和temporal transformer 部分，减少了训练耗时。更多的帧输入则保证了模型的精度。

## 3 Experiment

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/SFA-VIVIT2.png)
