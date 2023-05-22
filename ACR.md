# 自适应一致性正则化器（ACR）

> Wei T, Gan K. Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need[C]. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR). 2023: 3469-3478

## 一、引言

1. 长尾半监督学习（LTSSL）：在长尾分布中，少数类别的样本数量明显少于大多数类别的样本数量。

   传统的监督学习方法在这种情况下往往表现不佳，因为缺乏足够的标记数据来训练模型。同时在长尾数据集上训练的分类器往往会偏向多数类，从而导致对少数类的测试准确性较低。

   LTSSL 结合了**半监督**学习和**长尾分布**的特点，以减少对标记数据的依赖并提高分类性能。它的核心思想是利用少量的标记数据和大量的未标记数据来训练分类模型。通过使用**未标记数据**，可以引入额外的信息来帮助模型更好地理解整个**数据集的分布**，特别是少数类别的样本。

   在 LTSSL 中，通常会使用重新采样、重新加权、标签平滑、伪标签对齐、无监督的聚类算法或生成模型对未标记数据进行预处理，以获得额外的数据表示或生成新的标记样本。这些新的标记样本可以用于增强模型的训练，从而改善在长尾分布中的分类性能。

2. LTSSL 方法的缺陷：现有的 LTSSL 算法通常假设标记和未标记数据的**类分布相同**。当标记数据和未标记数据的类分布不匹配时，基于该假设构建的那些 LTSSL 算法可能会受到严重影响，因为它们使用来自模型的有偏见的伪标签。

3. 自适应一致性正则化器（ACR）：通过估计未标记数据的真实类别分布，实现了统一公式中各种分布的伪标签的动态精炼。

4. 相关工作：

   （1）DARP 和 CReST 通过**分布对齐**消除模型生成的有偏伪标签，以根据标记数据的类分布细化伪标签。

   （2）ABC 使用辅助平衡分类器，通过对多数类进行**下采样**来训练以提高泛化能力。

   （3）CoSSL 使用 mixup 为少数类设计了一个新的**特征增强模块**来训练平衡分类器。

   （4）DASO 通过基于未标记数据的当前估计类分布采用**线性**和**语义**伪标签的动态组合来处理标记和未标记数据的类分布不同。

5. 文章贡献：

   （1）提出了一个双分支网络，包括一个平衡分支和一个标准分支解决长尾分布的问题。

   （2）通过改进原始伪标签以匹配未标记数据的真实类别分布并提高其准确性。

   （3）在所有测试用例中改进了最近的 LTSSL 算法（DARP、CReST、DASO）。

## 二、自适应一致性正则化器（ACR）

1. 模型结构：

   <img src="images/162.png" alt="162" style="zoom: 50%;" />

2. 对于有标记的样本：

   （1）标准分支：交叉熵。用于学习优质的特征提取器。

   （2）平衡分支：使用平衡误差。用于学习类平衡的估计器。
   $$
   \mathcal{L}_{\text {b-labeled }}=-\sum_{i=1}^N \log \frac{e^{\tilde{f}_{y_i^{(l)}}\left(x_i^{(l)}\right)+\tau \cdot \log \pi_{y_i^{(l)}}}}{\sum_{c=1}^C e^{\tilde{f}_c\left(x_i^{(l)}\right)+\tau \cdot \log \pi_c}}
   $$

3. 对于无标记的样本：偏向于少数类的伪标签可以有利于分类器的学习，而近似真实分布的伪标签分布有助于学习更好的特征提取器。

   先使用标准分支提取特征，得到伪标签。再将其与增强后的平衡分支进行一致性最小化损失。
   $$
   \tilde{q}\left(x_j^{(u)}\right)=\arg \max f\left(x_j^{(u)}\right)-\tau \cdot \log \pi\\
   \mathcal{L}_{\mathrm{b} \text {-con }}=\sum_{j=1}^M \tilde{M}\left(x_j^{(u)}\right) \ell\left(\tilde{f}\left(\mathcal{A}\left(x_j^{(u)}\right)\right), \tilde{q}_j\right)
   $$
   同时使用平衡分支进行未标记数据集的类自适应优化。通过双向克里散度平衡分布，自适应消除多数类的偏见。
   $$
   \text {dist}_{\mathrm{con}}=\frac{1}{2}\left(D_{K L}\left(\pi_{\mathrm{con}} \| \pi_{\mathrm{est}}\right)+D_{K L}\left(\pi_{\mathrm{est}} \| \pi_{\mathrm{con}}\right)\right) \\
   \operatorname{dist}_{\mathrm{uni}}=\frac{1}{2}\left(D_{K L}\left(\pi_{\mathrm{uni}} \| \pi_{\mathrm{est}}\right)+D_{K L}\left(\pi_{\mathrm{est}} \| \pi_{\mathrm{uni}}\right)\right) \\
   \text {dist}_{\mathrm{rev}}=\frac{1}{2}\left(D_{K L}\left(\pi_{\mathrm{rev}} \| \pi_{\mathrm{est}}\right)+D_{K L}\left(\pi_{\mathrm{est}} \| \pi_{\mathrm{rev}}\right)\right)\\
   \tau(t)=\frac{2 e^{d i s t_{\mathrm{con}}^{(t-1)}}}{e^{d i s t_{\mathrm{con}}^{(t-1)}}+e^{d i s t_{\mathrm{uni}}^{(t-1)}}+e^{d i s t_{\mathrm{rev}}^{(t-1)}}}
   $$
   ![163](images/163.png)

## 三、结果

![164](images/164.png)