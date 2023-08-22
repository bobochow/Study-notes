# Multimodal Adaptation of CLIP for Few-Shot Action Recognition

> Xing, Jiazheng, et al. "Multimodal Adaptation of CLIP for Few-Shot Action Recognition." arXiv preprint arXiv:2308.01532 (2023).

## 1. Motivation

- CLIP-FSAR微调了整个CLIP模型，由于少样本训练数据的局限。这可能会导致过拟合的问题，并且增加训练的开销。
- 此外，CLIP-FSAR仅采用了一个简单的时序模块将图像预训练模型迁移到视频领域，这会导致其对于SthV2这类的数据集表现不好。
- 因此，本文采用了增加额外的Adapter模块的parameter-efficient fine-tuning，减少了训练参数。还设计了更复杂的文本-时序建模模块，增强了视频原型的表示。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP1.png)

### 2.1 Architecture Overview

- Task-oriented Multi-modal Adaptation模块在加入常规的adapter的同时引入了文本特征。本文的Adapter结构与AIM类似。
- Text-guided Prototype Construction Module模块在使用注意力机制进行时序建模的同时，也引入了文本特征，增强视频原型的表示。
- 最后将增强后的特征输入原型匹配度量得到预测的概率分布。

### 2.2 Task-oriented Multi-modal Adaptation

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP2.png)

- 通过改变输入数据的维度，本文共有3种Adapter结构，但其中自注意力权重是一致的。

#### 2.2.1 Temporal Adaptation

- 为了减少计算量，Temporal Adaptation的输入只有class token。此外，该Adapter中没有残差连接，原因是为了在训练初始阶段消除时序适应的影响。

#### 2.2.2 Multi-modal Adaptation

- 该模块将文本特征和时空特征进行了融合

#### 2.2.3 Joint Adaptation

- 通过平行的adapter微调ViT模块最后的输出表示。
- 为了保证空间维度的一致性，需要先进行select操作，选择Multi-modal Adaptation或Spatiotemporal Adaptation前N+1的特征。

### 2.3 Text-guided Prototype Construction Module

- 在少样本识别中，类别原型构建的质量将直接影响类别原型匹配的结果。而文本可以提供更多的语义信息，因此本文采用交叉注意力机制，将文本特征引入到原型构建中，增强了视频原型的表示。

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP3.png)

### 2.4 Metric Loss and Predictions

- 过去少样本动作识别通常采用时序对齐或原型的方法，但借助多模态的CLIP模型，现在我们可以借助任务文本特征辅助查询样本的分类。
- 为了使视频特征与文本特征更加接近，本文计算了全局池化的视频特征与文本特征的余弦相似度，并采用KL散度作为损失函数。
- 最终的损失函数和预测结果也都是由两部分组成：

$$\mathcal{L}=\alpha\cdot\frac12\left(\mathcal{L}_{\mathcal{S}2\mathcal{T}}+\mathcal{L}_{\mathcal{Q}2\mathcal{T}}\right)+(1-\alpha)\cdot\mathcal{L}_{\mathcal{Q}2\mathcal{S}}$$

$$\mathbf{p}=\alpha\cdot\mathbf{p}_{\mathcal{Q}2\mathcal{T}}+(1-\alpha)\cdot\mathbf{p}_{\mathcal{Q}2\mathcal{S}}$$

## 3. Experiment

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP6.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP7.png)

![8](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP8.png)

![9](https://raw.githubusercontent.com/bobochow/blog_img/main/img/MA-CLIP9.png)
