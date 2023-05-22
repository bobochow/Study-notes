# Rethinking Efficient Tuning Methods from a Unified Perspective

> Jiang, Zeyinzi, et al. "Rethinking Efficient Tuning Methods from a Unified Perspective." arXiv preprint arXiv:2303.00690 (2023).

## 1. Motivation & Contribution

### 1.1 Motivation

- 现有的模型调优方法，如prompt、prefix和adapter，对原始模型结构的不同部分执行特定于任务的轻量级调整，而原始模型的其他部分则没有涉及到。这导致现有的PETL方法改变了部分模块的数据分布，但这给剩余的冷冻模型部件适应新的分布带来了困难。

![1](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning1.png)

- 此外，现有PETL方法百花齐放，但其本质离不开冻结参数的操作模块和轻量化可训练模块。之前的PETL方法并没有提出针对这些不同的操作提出一个统一的框架。

### 1.2 Contribution

- 本工作从统一的角度对现有参数高效迁移学习方法（Parameter-efficient Transfer Learning, PETL）进行重新思考。一方面，进一步审视了现有的调优范式，提出了主流调优方法的并行化形式，以降低了模型结构的耦合度。另一方面，为参数高效的迁移学习提供了一个统一的框架，称之为U-Tuning（Unified Tuning）。
- U-Tuning由具有冻结参数的操作（OP）和统一的轻量化可训练结构（U-Tuner）组成（见下图），该框架允许灵活插入或移除可训练的调优结构，不仅可以覆盖大多数现有方法，还可以推导出新的调优结构。该框架具备足够的通用性，并且派生的新结构在各种下游任务上实现了相当或更好的性能。

## 2. Method

### 2.1 Prefix, Prompt, and Adapter tuning

![2](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning2.png)

#### 2.1.1 Prefix tuning

$$\mathrm{MHA}_{\text {pre }}=\operatorname{Attn}\left(\boldsymbol{x} \boldsymbol{W}_q,\left[\boldsymbol{K}_{\text {pre }} ; \boldsymbol{x} \boldsymbol{W}_k\right],\left[\boldsymbol{V}_{\text {pre }} ; \boldsymbol{x} \boldsymbol{W}_v\right]\right)$$

#### 2.1.2 Prompt tuning

$$\operatorname{MHA}_{\text {pro }}=\operatorname{Attn}\left(\left[\boldsymbol{x} ; \boldsymbol{x}_{p r o}\right] \boldsymbol{W}_q,\left[\boldsymbol{x} ; \boldsymbol{x}_{p r o}\right] \boldsymbol{W}_k,\left[\boldsymbol{x} ; \boldsymbol{x}_{p r o}\right] \boldsymbol{W}_v\right)$$

#### 2.1.3 Adapter tuning

$$\mathrm{FFN}_{\text {adapter }}=\underbrace{\operatorname{FFN}(\boldsymbol{x})}_{\text {original module }}+\underbrace{\phi\left(\mathrm{FFN}(\boldsymbol{x}) \boldsymbol{W}_{\text {down }}\right) \boldsymbol{W}_{u p}}_{\text {adapter module in parallel }}$$

### 2.2 U-Tuning

![3](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning3.png)

$$x ^ { \prime } = O P ( x ) + U - \operatorname { T u n e } x ( x )$$

- U-Tuning框架将统一公式中的Transformer的每个部分视为一个具有冻结预训练参数的操作函数OP，而每个调优部分则视为一个具有可学习参数的统一调优器U-Tuner。
- 当我们用类似的操作实例化OP和U-Tuner时，该公式覆盖所有现有的调优方法。同时，当我们用不同的构建模块实例化它们时，可以组合生成新的参数高效迁移方法。此外，与调优结构仅附加到操作子集的现有调整方法相比（如仅附加到MHA或仅附加到FFN），本方法可以将U-Tuner附加到所有操作（MHA和FFN）或甚至是Transformer Block。

## 3. Experiment

![4](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning4.png)

![5](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning5.png)

![6](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning6.png)

![7](https://raw.githubusercontent.com/bobochow/blog_img/main/img/utuning7.png)
