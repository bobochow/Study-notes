# FlowNet

> Dosovitskiy A, Fischer P, Ilg E, et al. Flownet: Learning optical flow with convolutional networks[C]. Proceedings of the IEEE international conference on computer vision (ICCV). 2015: 2758-2766.

> Ilg E, Mayer N, Saikia T, et al. Flownet 2.0: Evolution of optical flow estimation with deep networks[C]. Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR). 2017: 2462-2470.

## 一、引言

1. FlowNet 是早期使用深度学习方法预测光流的案例，属于是对行为识别输入的优化。对于光流预测，一方面是要“高质量”作为较优的输入，另一方面是要“高效率”以解决双流网络实时性的问题。

   高质量的光流算法：EpicFlow、DeepFlow、FlowFields、LDOF（CPU、GPU）、PCA-Layers。

   高效率的光流算法：EPPM、PCA-Flow、DIS-Fast、**FlowNetS**、**FlowNetC**。

2. 光流预测既需要每个像素的精准定位，也需要两帧图像的对应关系。FlowNet 使用相关层（correlation layer）获取帧间不同位置的相关性。

3. 光流预测算法很难同时捕获大位移和小位移，通用的方法是“由粗到细”。FlowNet 将这种方法应用到深度学习，先使用卷积获取低分辨率特征图捕获大位移，再使用上卷积细化到高分辨率捕获小位移。但“由粗到细”的通病在于如果在低分辨率下产生错误将很难在高分辨率下恢复，产生大量噪点且无法捕获大位移的小物体（比较严苛的条件）。于是 FlowNet 2.0 中额外添加了 FlowNetSD 一条流，单独考虑高分辨率下的小位移。

4. 深度学习建模需要大量的训练样本，可以使用数据增强但效果不佳。于是 FlowNet 的作者合成了一个 FlyingChairs 数据集（简称 Chairs 或者 C，合成的，椅子在天上飞）。但由于是合成数据集，与现实的动作差异较大，导致模型在 Sintel 和 KITTI 上的泛化能力较差。

## 二、FlowNet

1. FlowNetS（Simple）：将两帧图像直接叠加，使用通用网络，让网络自行决定如何处理图像以提取运动信息。

2. 细化（Refinement）：使用上卷积（类似于matlab的 .\* 操作，1\*1变成5\*5）将粗特征图细化到高分辨率捕获小位移。

3. FlowNetC（Correlation）：为两个图像创立两个独立但相同的处理流，并在特征提取后将其融合。

   网络先分别生成两张图像的特征表示，然后在更高的层次上组合它们，类似于标准匹配的方法。这种方法在本文实验中不如 FlowNetS ，但在 FlowNet2.0 中被发现在更复杂的场景中优于 FlowNetS。

4. 相关层：在第三个维度上求积W\*H\*D的特征图在第二帧半径为d的邻域中求相关，得到W\*H\*D\*D的四维相关性（D=2d+1）。
   $$
   c(x_1,x_2)=\sum_{o\in[-k,k]\times[-k,k]}<f_1(x_1+o),f_2(x_2+o)>
   $$

## 三、FlowNet2.0

1. FlowNet 虽然在效率上存在优势，但在精度上表现不佳，且难以检测小位移，容易产生噪声伪影。

2. FlowNet2.0 通过堆叠 FlowNetC 和 FlowNetS 提升模型精度，同时使用 FlowNetSD 检测小位移。

3. 在模型堆叠方面，结论如下：

   （1）单纯堆叠网络在 Chairs 上产生更好的效果，但在 Sintel 上会出现过拟合，效果更差。

   （2）扭曲能改善效果。

   （3）在 Net1 后添加中间损失有利。

   （4）最好的结果是保持第一个网络固定，只在扭曲后训练 Net2。

4. 扭曲（Warp）：把I2加上流场w=(u,v)，用于评估先前的错误同时方便计算增量。

5. FlowNetSD（Small Displacement）：用于检测小位移。在网络开始时使用更小的kernel，同时在upconv之间加入conv使更平滑。

## 四、结果