# What Can Simple Arithmetic Operations Do for Temporal Modeling?

> Wu, Wenhao, et al. "What Can Simple Arithmetic Operations Do for Temporal Modeling?." arXiv preprint arXiv:2307.08908 (2023).

## 1. Motivation

- 过去一些工作通过计算帧间RGB差值建模时序运动，这种以一帧作为anchor，与另一帧成对地建模时序关系的方法蕴含着专家知识且无需额外参数。本文认为可以进一步挖掘相邻两帧间做差这种简单运算操作的潜力。
- 本文认为加减乘除等操作已经被用于通常的视觉和时序任务，并且具备可解释性。加法常被用于实现积分以获取图像中的累积或平均特征。而减法则能体现随时间的变化且能近似运动的趋势。乘法可以体现两帧间的相似性或相关性，除法与乘法相反能体现相邻两帧间变化剧烈的特征。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ATM1.png)

## 2 Method

- 本文将经过加减乘除后得到的特征视作对骨干网络的补充辅助特征，因此可在CNN和ViT模型中即插即用。

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ATM2.png)

### 2.1 Arithmetic Temporal Module (ATM)

- ATM模块首先通过一个Context Spanning模块为每帧构建一个Z帧的上下文帧邻域。其中上下文领域Z为1时上下文帧为下一帧，Z为2时上下文帧包含前一帧和后一帧，Z=4依此类推。
- 然后将每帧作为anchor frame 与上下文帧做逐像素的四则运算并将结果Concat。
- 接着使用3x3的Conv2D提取$C\times H \times W$空间特征，$T\times C$ 矩阵已包含了每帧的上下文时序线索。
- 最后，由于骨干网络是一个时序无关的模型，而ATM提取的辅助特征是时序相关的，二者存在Domain Gap。因此需要将辅助特征先将ZC维度重组合并，并用Conv2D减少通道数，与骨干网络特征保持一致。以残差连接的方式融入骨干网络，为时序无关的骨干模型提供时序线索。

### 2.2 Arithmetic Operations

- Addition:能捕捉特征表示的空间占用分布

$$\operatorname{Addition}( A , B )_{ ( c , h , w ) }= A ( c , h , w ) + B ( c , h , w )$$

- Subtraction:能近似估计光流，Concat后可以推理变化的平滑程度

$$\operatorname{Subtraction}( A , B ) _ { ( c , h , w ) } = A ( c , h , w ) - B ( c , h , w )$$

- Multiplication:两帧间的空间乘法可以捕捉空间特征的外观相似性。本文并没有采用全局乘法，而是采用局部领域内乘法，得到一个$P^2\times H \times W$的乘法tensor，最后使用Conv转换为$C \times H \times W$

$$
\begin{aligned}
\text { Multiplication }(A, B)_{(h, w)}= & A(h, w) \cdot B(h+i, w+j), \\
& (i, j) \in P \times P .
\end{aligned}
$$

- Division:除法与减法相似，不同之处在于除法标准化了。为了使训练过程稳定，采用了Log，$\epsilon$设为1避免Log出现负值。

$$
\operatorname{Division}(A,B)_{(h,w)}=\log{(A(c,h,w)+\epsilon)}-\log{(B(c,h,w)+\epsilon)}
$$

## 3 Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ATM3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ATM4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/ATM5.png)
