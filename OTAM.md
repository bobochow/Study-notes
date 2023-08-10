# Few-Shot Video Classification via Temporal Alignment

> Cao, Kaidi, et al. “Few-Shot Video Classification via Temporal Alignment.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, <https://doi.org/10.1109/cvpr42600.2020.01063>.

## 1. Motivation & Contribution

![0](https://raw.githubusercontent.com/bobochow/blog_img/main/img/OTAM0.png)

- 以前的方法往往直接采用时序平均特征来预测最终结果，但这会带来时序信息的缺失。在一般的动作识别方法中由于训练样本充足，视频帧顺序并未得到重视。但在小样本学习中，视频帧的时间顺序信息的重要性就显而易见了。本文提出了一种新的方法，通过对齐视频帧来捕获时序信息，从而提高小样本学习的性能。
- 本文提出了一种时序对齐度量方法，显式地利用非线性的视频帧时间顺序信息，提高了小样本动作识别的性能。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/OTAM1.png)

### 2.1 Embedding Module

- 使用一个图像预训练的CNN网络来提取视频帧的特征，得到一个$T\times D_f$的特征矩阵。
- 然后计算query video和support video的特征矩阵的余弦相似度，得到一个$T\times T\times n$的余弦距离矩阵。

$$D_{l,m}=1-\frac{f_\varphi(S_i)_{l,}\cdot f_\varphi(S_j)_{m,}}{||f_\varphi(S_i)_{l,}||||f_\varphi(S_j)_{m,}||}$$

- 定义了一个二值对齐矩阵$\mathcal{W}\subset\{0,1\}^{T\times T}$，当query video的第$i$帧和support video的第$j$帧对齐时，$W_{i,j}=1$，否则$W_{i,j}=0$。
- 模型最终的目标是找到query video 与support video 的最佳对齐方式，使得对齐矩阵$W$和余弦距离矩阵$D$的内积最小。

$$W^{*}=\underset{W\in\mathcal{W}}{\operatorname{argmin}}\langle W,D(f_\varphi(S_i),f_\varphi(S_j))\rangle $$

- 两个视频的距离度量（即图中的Alignment Score），如下表示：

$$\phi(f_\varphi(S_i),f_\varphi(S_j))=\langle W^*,D\rangle $$

- 本文通过Dynamic Time Warping (DTW)算法求解两个序列的相似度，得到一条最佳的对齐路径。
- 其思路是动态规划：当前元素的最短路径必然是从前一个元素的最短路径的长度加上当前元素的值。前一个元素只有三个可能，取三个可能之中路径最短的那个即可。

$$\gamma(i,j)=D_{ij}+\min\{\gamma(i-1,j-1),\gamma(i-1,j),\gamma(i,j-1)\}$$

- 由于两个视频中的动作并不一定能从头到尾完全对齐，因此本文做了修改，使得对齐路径的起点和终点更加灵活，而不局限于第一帧和最后一帧。本文在距离矩阵的开始和结束处填充两列0s，这样就可以从任意一帧开始，到任意一帧结束，得到一条最佳的对齐路径。

$$\begin{aligned} & \gamma(i,j)=\\  & D_{ij}+\begin{cases}\min\{\gamma(i-1,j-1),\gamma(i-1,j),\gamma(i,j-1)\}, & j=0 \space \mathrm{or}\space j=T+1\\  & \\ \min\{\gamma(i-1,j-1),\gamma(i,j-1),\mathrm{otherwise} & \end{cases}\end{aligned}$$

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/OTAM2.png)

- 但由于取最小操作不可导，因此本文采用log-sum-exp来近似估计最小值，使得整个过程可导。

$$\min(x_1,x_2,...,x_n)\approx-\lambda\log\sum_{i=1}^ne^{-x_i/\lambda}\mathrm{~if~}\lambda\to0.$$

$$\phi(f_\varphi(S_i),f_\varphi(S_j))=\gamma(T,T+1)$$

- 训练损失函数为：

$$\mathcal{L}=-\log\frac{\exp(-\phi(f_\varphi(S),f_\varphi(\hat{S})))}{\sum_{Z\in\mathcal{S}}\exp(-\phi(f_\varphi(S),f_\varphi(Z)))}$$

- 测试推理结果：
$$S^{*}=\underset{S\in\mathcal{S}}{\operatorname{argmin}}\phi(Q,S)$$

## 3. Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/OTAM3.png)
