# Multiscale Vision Transformers

> Fan, Haoqi, et al. “Multiscale Vision Transformers.” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2022, <https://doi.org/10.1109/iccv48922.2021.00675>.

## 1 Motivation & Contribution

- 多尺度特征从浅层到深层根据channel 的维度划分成多个stage，channel维度逐渐增大，空间分辨率逐渐变小。浅层可以在高空间分辨率下用小的channel维度建立简单的low-level feature，而深层则可以用更大的channel维度建立更high-level的语义信息。
- 提出了 multi head pooling attention，将transformer和多尺度特征结合起来，提高transformer模型在各种输入分辨率下的性能，同时大大降低了计算开销。
- 将CNN思想与Transformer结合起来

## 2 Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViT_1.png)

### 2.1 Multi Head Pooling Attention

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViT.png)

- 与常规的MHSA固定通道维度和时空分辨率不同，MHPA通过池化操作减小了token数量，从而减少了计算量。

### 2.2 Multiscale Vision Transformers

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViT_2.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViT_3.png)

#### 2.2.1 Decoupled pooling

- query pooling : 能减少输出的token长度
- Key-Value pooling : 能减少注意力计算复杂度
- 因此，只在每个stage的第一次做Query pooling,在所有层使用Key-Value pooling。

## 3 Experiment

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViT_4.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MViT_5.png)
