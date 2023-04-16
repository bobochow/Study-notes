# Revisiting Skeleton-based Action Recognition

> Duan, Haodong, et al. “Revisiting Skeleton-Based Action Recognition.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, <https://doi.org/10.1109/cvpr52688.2022.00298>.

---
> 参考论文解读：<https://zhuanlan.zhihu.com/p/395588459>

## 1 Motivation & Contribution

### 1.1 Motivation

- 大部分骨骼动作识别的工作采用 GCN 来提取骨骼的特征。尽管被广泛使用，但 GCN 方法依然在鲁棒性、兼容性和可扩展性上存在一定缺陷。
  - 鲁棒性： 输入的扰动容易对 GCN 造成较大影响，使其难以处理关键点缺失或训练测试时使用骨骼数据存在分布差异（例如出自不同姿态提取器）等情形。
  - 兼容性： GCN 使用图序列表示骨架序列，这一表示很难与其他基于 3D-CNN 的模态（RGB, Flow 等）进行特征融合。
  - 可扩展性：GCN 所需计算量随视频中人数线性增长，很难被用于群体动作识别等应用。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PoseC3D_1.png)

### 1.2 Contribution

- 提出了一个基于3D conv和3D 热图的骨骼动作识别新方法，解决了基于GCN方法的缺陷。
  - 3D热图更加鲁棒
  - 基于卷积网络设计更加灵活
  - 高效处理多人动作

## 2 Method

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PoseC3D_2.png)

### 2.1 Good Practices for Pose Extraction

- 人体骨架或姿势提取是基于骨架的动作识别的关键预处理步骤，在很大程度上影响最终的识别精度。
- 考虑到二维人体姿态具备更高的质量，作者选择了以二维人体姿态而非三维作为输入。
- 考虑到其在 COCO 关键点识别任务上的良好性能，作者使用了以 HRNet 为主干网络的 Top-Down 姿态估计模型作为姿态提取器。
- 模型的直接输出为关键点热图。在实践中，直接存储关键点热图会消耗大量磁盘空间。为提升效率，将每个 2D 关键点存储为坐标 (x, y, score)，其中 score 为预测的置信度。实验结果表明精度仅有少量下降。

### 2.2 From 2D Poses to 3D Heatmap Volumes

- 基于提取好的 2D 姿态，需要堆叠 $T$ 张形状为 $K\times H\times W$ 的二维关键点热图以生成形状为 $K\times T \times H\times W$ 的 3D 热图堆叠作为输入。若事先将 2D 姿态存储成坐标形式，则需要先借助生成以 $(x_i,y_i)$ 为中心，$c_i$为最大值的高斯分布，将其重新转换为热图形式。
- 在实践中，作者使用了两种方法来尽可能减少 3D 热图堆叠中的冗余，使其更紧凑。首先根据视频中人的位置，寻找一个最紧的框以包含所有帧中的所有人。在此之后，根据找到的框对每帧的热图进行裁剪，并将裁剪后的热图重新缩放至特定大小。借助这一方式，我们在空间上降低了冗余，在一个相对小的$H\times W$ 大小下包含了更多的信息。
- 此外，利用均匀采样以减少 3D 热图堆叠在时间维度上的冗余。由于整个视频长度过长，难以处理，通常选取一个仅包含部分帧的子集构成一个片段，作为 3D-CNN 的输入。基于 RGB 模态的方法，通常只在一个较短的时间窗内采帧构成 3D-CNN 的输入（如 SlowFast 在一个长仅为 64 帧的时间窗内采帧）。由于这种采帧方式难以捕捉整个动作，因此在骨骼行为识别中，采用了均匀采样的方式：需要采帧$N$帧时，先将整个视频均分为长度相同的$N$段，并在每段中随机选取一帧。在实验中，发现这样的采帧方式对骨骼行为识别尤其适用。

### 3D-CNN for Skeleton-based Action Recognition

- 在本工作中，作者基于骨骼模态和骨骼 + RGB 模态，分别设计了两种 3D-CNN： Pose-SlowOnly 与 RGBPose-SlowFast。Pose-SlowOnly 仅以骨骼模态作为输入。RGBPose-SlowFast，它包含两个分支，分别处理 RGB 和骨骼两个模态。RGB 分支具有低帧率以及更大的网络宽度，骨骼分支具有高帧率和更小的网络宽度。两分支间存在双向连接，以促进模态间的特征融合。将两分支的预测结果融合，作为最终的预测。在训练时，用两个单独的损失函数分别训练两个分支，以避免过拟合。

## 3 Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PoseC3D_3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PoseC3D_4.png)
