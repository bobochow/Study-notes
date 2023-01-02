# PWC-Net

> Sun D, Yang X, Liu M Y, et al. Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume[C]. Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 8934-8943.

## 一、引言

1. FlowNet的缺点：

   （1）堆叠的大模型容易过拟合，子网络必须顺序训练。

   （2）内存占用大（640MB）。

2. PWC-Net的特点：紧凑（compact），比 FlowNet2.0 小 17 倍。

   !(images\71.png)

## 二、PWC-Net

!(images\72.png)

（左侧：传统c-t-f的结构。右侧：PWC-Net。）

1. 由于原始图像会受阴影、光照影响，采用可学习的特征金字塔代替固定图像金字塔。

2. 扭曲（wrapping）作为单独一层估计大运动。

   与 FlowNet2.0 相似，再 Image2 上加上流场 w 。但 FlowNet2.0 是在子网络间添加的扭曲，目的是评估先前的错误和方便计算增量。这里是在金字塔层间添加的扭曲，目的是在高分辨率中添加低分辨率下的信息（即大运动）。
   $$
   c_w^l(x)=c_2^l(x+up_2(w^{l+1})(x))
   $$
   最高层的w为0。

3. 使用“损失体”（Cost volume）作为输入，更具辨别力。

   “损失体”是一个像素点与下一帧像素间的匹配损失，用 I1 的像素点与 I2 半径为 d 的邻域内的相似度表示。
   $$
   cv^l(x_1,x_2)=\frac{1}{N}(c_1^l(x_1))^Tc_w^l(x_2)
   $$
   N是特征图的通道维度。最终得到的“损失体”维度为d\*d\*H\*W。

4. wrapping 和 cost volume 层没有学习参数，因此减小了模型大小。

5. 光流估计：一系列的CNN层。

6. 上下文网络：使用膨胀卷积有效扩大感受野的大小。

   在传统c-t-f方法中，通常采用中值滤波或双边滤波结合上下文信息后处理改进光流。

## 三、结果

!(images\73.png)