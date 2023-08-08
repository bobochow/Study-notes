# CLIP-guided Prototype Modulating for Few-shot Action Recognition

> Wang, Xiang, et al. "CLIP-guided prototype modulating for few-shot action recognition." arXiv preprint arXiv:2303.02982 (2023).

## 1 Motivation & Contribution

- 尽管如OTAM、STRM、HyRSM等工作在少样本动作识别任务上取得了很好的效果，但这些模型受限于有限的标注数据和缺乏多模态信息，因此将如CLIP模型这样的多模态基础模型应用于少样本动作识别任务中仍有待探索。
- 由于存在domain gap无法直接将CLIP模型应用于少样本动作识别任务中,且若简单将CLIP模型作为初始化模型进行微调，则由于未有效利用多模态信息实际效果不佳。因此，本文基于CLIP模型提出了一种新的少样本动作识别方法，通过引入文本信息，增强了模型少样本识别能力。
- 为了让CLIP适应少样本视频动作识别，本文设计了一个视频文本对比优化目标，通过优化该目标，使得视频和文本的特征在嵌入空间中更加接近。
- 此外，还通过temporal Transformer将文本特征和视频特征融合，以便为视觉原型引入文本信息。

## 2 CLIP-FSAR

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/CLIP-FSAR.png)

### 2.1 Video-text contrastive objective

- 首先使用CLIP模型的image encoder和text encoder分别提取视频和对应类别文本的特征，然后模拟CLIP训练目标，通过最大化视频和文本特征的相似性来优化模型。
- 计算视频特征和文本特征的匹配概率分布，然后通过交叉熵优化目标。
$$p_{(y=i|v)}^{video-text}=\frac{\exp(\operatorname{sim}(\mathrm{GAP}(f_v),w_i)/\tau)}{\sum_{j=1}^B\exp(\operatorname{sim}(\mathrm{GAP}(f_v),w_j)/\tau)}$$

### 2.2 Prototype modulation

- 现有的少样本动作识别方法通常通过计算query video与support prototype的时序对齐距离进行分类，因此模型性能受prototype的估计精度影响较大。
- 少样本条件下，视觉信息欠缺导致原型的估计不准确，因此本文提出了一种基于文本信息的原型建模方法，通过引入文本先验信息，增强了模型少样本识别能力。
- 本文通过temporal Transformer将文本特征和视频特征融合，由于测试时无法得知query video的类别，因此测试时仅将视频特征输入到temporal Transformer中，然后依据时序对齐度量计算query video与support prototype的时序对齐距离。

$$d'_{q,s_i}=\mathcal{M}(\widetilde{f}_q,\widetilde{f}_{s_i})$$

- 基于时序对齐距离计算query video与support prototype的概率分布，然后通过交叉熵优化目标。
$$p_{(y=i|q)}^{few-shot}=\frac{\exp(d'_{q,s_i})}{\sum_{j=1}^N\exp(d'_{q,s_j})}$$

- 最终的损失函数则是两个损失函数的加权和。

$$\mathcal{L}=\mathcal{L}_{video-text}+\alpha\mathcal{L}_{few-shot}$$

- 对于query video 也可以结合两个概率分布进行预测类别。

$$p_{(y=i|q)}^\dagger=(p_{(y=i|q)}^{video-text})^\beta\cdot(p_{(y=i|q)}^{few-shot})^{1-\beta}$$

## 3 Experiments

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/CLIP-FSAR2.png)

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/CLIP-FSAR3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/CLIP-FSAR4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/CLIP-FSAR5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/CLIP-FSAR6.png)
