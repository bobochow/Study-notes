# A Unified Pyramid Recurrent Network for Video Frame Interpolation

> Jin, Xin, et al. "A Unified Pyramid Recurrent Network for Video Frame Interpolation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

## 1. Motivation & Contribution

- 将双向光流估计和中间帧生成集成到一个轻量化的循环图像金字塔架构中，通过金字塔递归修正，使得轻量化模型也能取得很好的效果。

## 2. Recurrent Frame Interpolation Modules

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/upr_net1.png)

- 每层金字塔都由一个特征编码器提取多尺度的图像特征，编码器最后一层提取的特征与上一层金字塔上采样后得到的粗粒度光流估计输入双向光流估计模块得到细化的光流估计。
- 细化的双向光流估计与当前尺度图像、多尺度CNN特征经forward warp后，结合上层上采样的中间帧插帧生成细化的中间帧插帧图像。

### 2.1 Feature encoder

- 由于forward warp存在伪影问题，所以为了鲁棒地生成中间帧，主流做法都是在中间帧生成时加入额外的图像特征，使得模型能够从图像特征中推断出中间帧图像。
- 本文采用了一个三阶段多尺度图像编码器，最后一层的特征会用于双向光流估计，所有尺度的特征会用于中间帧合成。

### 2.2 Bi-directional flow module

- 首先使用金字塔上层上采样后得到的粗粒度光流作为初始化，然后结合CNN最后一层的特征forward warp得到中间帧扭曲特征。接着使用中间帧扭曲特征构建correlation volume，最后将correlation volume、扭曲特征和初始化光流一起输入CNN光流预测器得到细化的光流估计。

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/upr_net2.png)

### 2.3 Frame synthesis module

- 帧合成模块基于一个U-Net结构，encoder的输入为forward warp 后的输入帧、对应多尺度特征以及上一层上采样后的粗粒度光流估计。
- 最后根据输出map和对应光流即可生成中间帧。

$$
I_t^l=\frac{(1-t) \cdot M_0^l \odot I_{0 \rightarrow t}^l+t \cdot M_1^l \odot I_{1 \rightarrow t}^l}{(1-t) \cdot M_0^l+t \cdot M_1^l}+\Delta I_t
$$

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/upr_net3.png)

## 3. Experiment

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/upr_net4.png)
