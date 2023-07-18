# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

> Ryali, Chaitanya, et al. "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles." arXiv preprint arXiv:2306.00989 (2023).

## 1. Motivation & Contribution

- 目前主流的模型（Swin, MViT）大多将卷积模型中的多层次思想与ViT结合，虽然能够取得不错的精度和计算效率，但是由于复杂的多层级结构，其推理速度要慢于原始ViT模型。这与先前以ResNet为代表的多层次模型的表现相反。
- Swin等模型通过在transformer中引入卷积、滑动窗口或相对位置编码来弥补原始transformer中缺失的归纳偏置，但是这些操作都会减慢模型推理速度。
- 而在自监督学习中流行的MAE通过设置mask-reconstruct这样的视觉代理任务，使得模型自身能够学习到空间推理。
- 因此，本文认为将多层次思想与掩码重建自监督学习任务结合，抛弃那些复杂冗余的操作，模型仅采用纯ViT结构即可得到精确且推理快速的多层次ViT。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera1.png)

### 2.1 MAE for Hierarchical Models

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera2.png)

- 由于掩码操作会破坏空间布局，无法直接用于多层次模型。因此，Hiera首先将掩码单元的尺寸放大，将掩码单元与其他token区分开。然后有两张方法进行多尺度的下采样。首先是将可见单元视作独立区域，分别进行卷积等操作，缺点是需要padding会带来不必要的开销。第二种方法为将核的大小与步长保持一致，更加简单高效。

### 2.2 Mask Unit Attention

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera3.png)

- MViT中采用了pooling attention，即将K,V先做池化再做自注意力操作，减少了计算量，但这对于大型输入（如视频）来说可能很昂贵。
- 因此Hiear提出使用mask unit attention 取代pooling attention，仅在mask unit 中进行局部自注意力。其中mask unit的窗口尺寸会随下采样分辨率变化，而Swin中的窗口自注意力的窗口大小是固定的。

### 2.3 Simplifying MViTv2

- 在使用MViTv2进行MAE自监督训练时删去或简化其中不必要的部分依然能够保持较高的准确率。

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera4.png)

### 2.4 Configuration for Hiera variants

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera5.png)

## 3. Experiment

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera7.png)

![8](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera8.png)

![9](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera9.png)

![10](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera10.png)

### 3.1 MAE Ablations

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/hiera6.png)
