# How can objects help action recognition?

> Zhou, Xingyi, et al. "How can objects help action recognition?." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

## 1. Motivation & Contribution

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit1.png)

- 目前主流的视频模型大多将视频看作完整的长序列的spatio-temporal tokens，忽视了人与物体间的互动，一次处理所有的tokens。即使一些模型尝试丢弃一部分冗余的tokens，但会损失精度。
- 物体的运动和动作对得到精简的视频特征表示是有益的，也符合人类视觉感官优先捕捉关键区域的特点。因此，本文希望通过由物体引导token采样策略，有效减少token数量的同时显著不影响精度。还希望利用物体信息丰富视频特征表示，提高模型精度。

## 2. Method

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit2.png)

- ObjectViViT由两部分组成，分别是object-guided token sampling strategy (OGS)和object-aware attention module (OAM)。
- ObjectViViT相较于一般视频模型增加了目标检测的额外输入，OGS会根据检测到的物体位置判断某一Token属于前景还是背景，并且会保留前景token，丢弃背景token。
- 为了充分利用物体和时空块间的关系，OAM先将来自于同一物体的token分组，然后再与其他patch tokens做自注意力。

### 2.1 Object-guided Token Sampling

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit3.png)

- OTS使用CenterNet在每一帧生成一个类别无关的目标中心点热图，并计算每个token在热图中的分数，用以衡量token距离物体的远近。根据得分，选择前X%的tokens作为前景tokens，并在背景tokens中均匀采样Y%个，而剩余的则被丢弃。

$$H _ { t , i , j } = \operatorname{m a x e x p} ( - \frac { ( i - x _ { t , 0 } ) ^ { 2 } + ( j - y _ { t , 0 } ) ^ { 2 } } { 2\sigma^ { 2 }_ { t , 0 } } )$$

### 2.2 Object-aware Attention Module

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit4.png)

- 每个实例的热图代表了物体与其他token的相关性，因此对经OGS后的tokens针对不同的物体实例进行加权求和池化得到object token,最后将所有Object tokens 和其他tokens concat 作为keys和values，增强时空特征表示。

$$w _ { 0 , t } ^ { l } = \operatorname{M a x P o o l} ( \operatorname{MLP}(\hat{\mathbf{H}}_{o,t} * \hat{\mathbf{z}}_{t}^{l}) )$$

$$\mathcal{y} ^ { l } =\operatorname{ M H A} ( z ^ { l } , [ z ^ { l } , w ^ { l } ] , [ z ^ { l } , w ^ { l } ] ) + z ^ { l }$$

## 3. Experiment

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit6.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit7.png)

![8](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit8.png)

![9](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit9.png)

![10](https://raw.githubusercontent.com/bobochow/blog_img/main/img/objectvivit10.png)

## 4 Future Work

- 在人类行为交互数据集上训练的特定领域检测器效果最好,切换到更强的检测骨干或更通用的开放世界检测器并没有帮助,目前尚不清楚目标检测指标(即大多数数据集上的mAP)如何与帮助动作识别的能力相关联。
