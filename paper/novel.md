# Novel（新奇论文）

- [Neural Ordinary Differential Equations](https://www.jiqizhixin.com/articles/122302),[@Paper](https://arxiv.org/abs/1806.07366),来自机器之心

>本文主要介绍神经常微分方程背后的细想与直观理解，很多延伸的概念并没有详细解释，例如大大降低计算复杂度的连续型流模型和官方 PyTorch 代码实现等。这一篇文章重点对比了神经常微分方程（ODEnet）与残差网络，我们不仅能通过这一部分了解如何从熟悉的 ResNet 演化到 ODEnet，同时还能还有新模型的前向传播过程和特点。
其次文章比较关注 ODEnet 的反向传播过程，即如何通过解常微分方程直接把梯度求出来。这一部分与传统的反向传播有很多不同，因此先理解反向传播再看源码可能是更好的选择。值得注意的是，ODEnet 的反传只有常数级的内存占用成本。

