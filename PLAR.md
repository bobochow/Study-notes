# Prompt Learning for Action Recognition

> Wang, Xijun, et al. "Prompt Learning for Action Recognition." arXiv preprint arXiv:2305.12437 (2023).

## 1. Motivation & Contribution

- 以往的prompt learing提出的prompt主要用于image level，针对动作识别这样的视频领域的prompt仍有待研究。
- 本文进一步探索了用于动作识别的 prompt learning 方法，提出了使用更种类的prompt用来提升模型性能，包括光流、SAM、可学习prompt。
- 还提出了一种根据输入无关的专家prompt结合输入数据，生成依赖输入数据的可学习prompt方法，提高了模型域适应能力。

## 2. Method

![0](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PLAR.png)

### 2.1 Prompt Learning based Input Encoder

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PLAR1.png)

#### 2.1.1 Non-Learnable Prompt

- **Optical Flow Prompt**

- **Large Vision Model Prompt**
  - SAM

#### 2.1.2 Learnable Prompt

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PLAR2.png)

- 不同的视频有不同的动作和域(不同的视频源)，因此学习能用于所有视频的单一通用提示是具有挑战性的。因此，本文针对不同视频域设计了与输入无关的不同的专家prompt，用于为模型提高特定任务的信息。
- 结合专家prompt并赋予依赖输入数据的权重，不同的输入能得到不同的prompt。

### 2.2 Auto-regressive Temporal Reasoning

- 上述的prompt依然局限于2D空间域，对于时序推理的帮助较少，因此作者提出使用一个自回归时序推理算法更好地建模视频。
- 自回归模型是基于先验观察进行预测的模型，对于动作识别这样一段视频只包含一个动作类别的任务来说是有帮助的

## 3. Experiment

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/PLAR3.png)

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/202307201651511.png)
