# Knowledge Prompting for Few-shot Action Recognition

> Shi, Yuheng, Xinxiao Wu, and Hanxi Lin. "Knowledge prompting for few-shot action recognition." arXiv preprint arXiv:2211.12030 (2022).

## 1. Motivation

- 如CLIP-FSAR,MA-CLIP这类Parameter-efficient fine-tune工作，对于文本的处理通常只是简单采用了只涉及到动作类别的hard prompt，并没有设计更加复杂的text prompt。
- 本文利用外部语料库收集和生成了很多涉及人物动作的文本提示，通过这些丰富的文本提示来生成动作的语义特征，提升了模型的泛化能力。

## 2. Method

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/kprompt4fsar1.png)

- 本文并没有采用元学习或者微调基于分类器骨干网络的方法，而是将CLIP模型冻结，仅改变了CLIP模型text encoder的输入text prompt，唯一需要训练的部分是最后的融合多帧动作语义特征的时序建模模块。

### 2.1 Generation of Text Proposals

- 本文最重要的部分就是如何生成丰富的文本提示，这里采用了两种方法：
    1. 通过固定句式手工生成
    2. 通过文本提示网络自动生成

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/kprompt4fsar2.png)

#### 2.1.1 Handcraft Generation via Sentence Template

- 本文使用主谓宾形式的手工文本特征形式。由于没有直接描述动作的语料库，因此本文从与动作相关的数据集中收集相关文本。本文利用PaStaNet数据集中细粒度的身体部位运动状态概念，以及Visual Genome数据集中丰富的物体类别，生成了形如以下句子的文本提示：
  - Human's **[body part] [state]** the **[object]**

- **PaStaNet**标注格式：
![PaStaNet](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PaStaNet1.png)

- 但是由于初始的文本提示来自两个数据集，因此需要对文本进行清洗，去除一些不合理的文本提示。本文使用了一个预训练的掩码语言模型BERT来判断文本提示的合理性，即将object掩码，然后使用BERT计算掩码名词的概率，如果概率低于阈值，则认为该文本提示不合理。
- 经过过滤后就得到了动作文本知识库的主体部分了。

#### 2.1.2 Automatic Generation via Text Proposal Network

- 为了进一步扩大动作知识库的规模，本文采用了一个文本提示网络，从网络动作教学视频中提取动作文本提示。
- 首先，本文依据"how to"，"tutorial", "teach"这类搜索关键词，从YouTube上爬取了大量的跳水、健身等动作的网络教学视频中的字幕，并采用BIO的格式进行了数据标注，用于训练TPN。
- TPN由一个用于提取句子token特征的BERT模型和一个分类头组成。
- 本文将自动生成的本文提示分为两类，一类是描述整体运动的实例级描述，另一类是描述身体部位运动的细粒度描述。

### 2.2 Temporal Modeling of Action Semantics

- 为了建模不同帧间动作语义的时序关系，采用了一个轻量级的时序建模模块。

## 3. Experiments

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/kprompt4fsar3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/kprompt4fsar4.png)

## 4. Existed Problems

- 在SthV2数据集上效果一般，原因可能是模式时序建模模块过于简单，且SthV2数据集的类别粒度较小，难以收集到类似的文本提示。
