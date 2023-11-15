# Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone

> Jiang, Zeyinzi, et al. "Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone." arXiv preprint arXiv:2310.19859 (2023).

## 1. Motivation

- 继前作U-Tuing针对参数高效但显存不高效的问题，本文受Ladder Side Tuning的启发，提出了一种参数和显存都高效的旁路残差微调方法。

## 2. Method

![1](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/restuing1.png)

### 2.1 Unbinding Tuners from Foundation Models

- 本文依旧延续前作中提出的统一的参数微调模块设计，即将Adapter, Prefix, Prompt都修改成并行的结构。

- Prefix tuning:

$$\mathrm{MHA}_{\mathrm{pre}}=\mathrm{Attn}(\boldsymbol{xW}_q,[\boldsymbol{K}_{pre};\boldsymbol{xW}_k],[\boldsymbol{V}_{pre};\boldsymbol{xW}_v])$$

- Res-Prefix tuning:

$$\mathrm{MHA}_{\mathrm{pre}}=(1-\lambda)\underbrace{\mathrm{Attn}\left(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}\right)}_{\text{original attention}}+\lambda\underbrace{\mathrm{Attn}\left(\boldsymbol{Q},\boldsymbol{K}_{pre},\boldsymbol{V}_{pre}\right)}_{\text{prefix attention in parallel}},$$

$$\lambda(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{K}_{pre})=\frac{\sum_i\exp\left(\boldsymbol{Q}\boldsymbol{K}_{pre}^\top\right)_i}{\sum_i\exp\left(\boldsymbol{Q}\boldsymbol{K}^\top\right)_i+\sum_j\exp\left(\boldsymbol{Q}\boldsymbol{K}_{pre}^\top\right)_j},$$

- Prompt tuning:

![2](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/restuning2.png)

### 2.2 Res-Tuning-Bypass

- 通过detach骨干网络梯度的方式，减少显存开销，但缺点是精度下降，仍有待解决。

$$\begin{aligned}&x_0^\mathrm{bypass}=x_0,\\&x_l^\mathrm{bypass}=\lambda\text{Res-Tuner}(x_l)+(1-\lambda)\text{Res-Tuner}(x_{l-1}^\mathrm{bypass}),l\geq1\end{aligned}$$

## 3. Experiments

![3](https://cdn.jsdelivr.net/gh/bobochow/blog_img@main/img/restuning3.png)
