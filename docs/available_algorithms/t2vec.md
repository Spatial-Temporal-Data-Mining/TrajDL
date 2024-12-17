# T2VEC

[[ICDE2018]Deep representation learning for trajectory similarity computation](https://kaiqizhao.github.io/icde18-camera-ready.pdf)


## 摘要

计算*轨迹相似度*（trajectory similarity）是轨迹序列领域中的一项重要任务。传统算法一般是基于*动态规划*（dynamic programming）算法，这些方法存在如下三个主要的缺点：

  1. 时间复杂度$O(n^2)$
  2. 针对*非均匀*（non-uniformity）、*低采样率*（low sampling rate）以及*带噪声*（noisy）的轨迹序列效果不佳

因此，作者提出了基于seq2seq模型的深度学习方法**T2VEC**（**t**rajectory **to** **v**ector），旨在学习高质量的*轨迹嵌入表示*（trajectory embedding）以应对上述难点。该篇论文有如下贡献：

  1. 在没有样本标签的情况下，通过序列样本增强的方式来构建学习样本。
  2. 基于空间近邻的先验，对*噪声对比估计损失*（noise contrastive estimation）的正标签采样进行改进，提出了*空间近邻感知损失*（spatial proximity aware loss）

通过以上的方式，T2VEC能够更好的表征轨迹序列中位置距离相关的信息。


## 预备知识

seq2seq模型的结构如[图1](#seq2seq-arch)，输入是token序列 $\mathrm{x}$，输出是token序列 $\mathrm{y}$:

```{figure} ../_static/images/t2vec_seq2seq_model.jpg
:alt: seq2seq模型结构
:align: center
:width: 600px
:name: seq2seq-arch

图1. seq2seq模型结构（图源：[T2VEC](https://kaiqizhao.github.io/icde18-camera-ready.pdf)）
```

根据seq2seq模型的链式法则，可得到如下公式:

$$
\mathbb{P}(y_1,...,y_{|y|} \vert x_1,...,x_{|x|}) = \mathbb{P}(y_1 \vert x) \prod_{t=2}^{|y|} \mathbb{P}(y_t \vert y_{1:t-1},x)
$$ (chain_rule_1)

输入的轨迹序列$\mathrm{x}$经过*编码器*（encoder）输出为固定维度（如$d=512$）的向量表示$v$，在RNN系列的模型中即为编码器最后一个*单元*（unit）输出隐藏状态（hidden state），所以{math:numref}`chain_rule_1` 可以简化成如下:

$$
  \mathbb{P}(y_t \vert y_{1:t-1}, x) = \mathbb{P}(y_t \vert y_{1:t-1}, v)
$$ (chain_rule_2)

*解码器*（decoder）在每一个时间$t$上都会将$v$和$y_{1:t-1}$解码成隐状态$h_{t}$:

$$
  h_t = f_{dec}(v, y_{1:t-1}) = \begin{cases} f_{dec}(v, EOS) & t=1 \\ f_{dec}(h_{t-1}, y_{t-1}) &  t >= 2, \end{cases}
$$ (decoding)

所以，$\mathbb{P}(y_t \vert y_{1:t-1}, v)$可以被建模成:

$$
  \mathbb{P}(y_t=u \vert y_{1:t-1}, v) = \mathbb{P}(y_t=u|h_t)=\frac{\exp(W_u^Th_t)}{\sum_{v \in V}\exp(W_v^Th_t)}
$$ (post_prob)

$W^T$是从*隐状态空间*（hidden state space）映射到*词表*（vocabulary）的映射矩阵，$u$表示词表中的一个候选元素，$W_u^T$定义为该映射矩阵的第$u$行，$V$表示字典空间的大小。

综合上述公式{math:numref}`chain_rule_1`{math:numref}`chain_rule_2`{math:numref}`decoding`{math:numref}`post_prob`，显然，从输入$\mathrm{x}$经过seq2seq模型获得输出$\mathrm{y}$，在有监督学习的范式下，模型训练可以使用交叉熵损失。


## T2VEC算法


### 定义

在进行方法建模之前，先来做如下定义：

* $T$：*轨迹序列*（trajectory sequence），真实世界中采集到的离散轨迹点序列。
* $R$：*隐藏路径*（underlying route），*隐藏状态空间*（hidden state space）中的连续分布，从该分布中采样可以得到一条真实世界中的离散轨迹序列。该分布无直接的解析解（即我们无法写出该分布的数学公式），我们仅能够观测到该分布在真实世界中的采样——轨迹序列的集合。
* $P_\theta(T|R)$：先验分布，直接对该分布采样即可获得隐藏路径$R$的一条轨迹序列样本$T$
* $Q_\phi(R|T)$：后验分布，该分布基于轨迹序列样本$T$来推理该样本在隐藏状态空间中的分布$R$，该分布也没有解析解。

```{figure} ../_static/images/latent_space.svg
:alt: 隐藏路径建模
:align: center
:width: 300px
:name: latent-model

图2. 隐藏状态空间建模
```

```{tip}
此类建模方式已经是深度学习较为常用的建模方法，如果需要更进一步的了解，可以参见[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)。
```

在上面的定义中，认为$R$是$T$在连续的向量空间中的压缩表示，如果能够有一个模型能够对这个隐藏状态分布有较好的建模，那么$R$就是$T$较好的*向量表征*（vector representation）。假如这个模型已经得到了，那么输出轨迹$T$就可以拿到$T$对应的向量表示$R$。

对于此类轨迹序列相关的表征问题，很容易想到使用RNN系列的模型，但此处的关键是如何进行模型的训练。


### 方法建模

T2VEC算法的目标是给定轨迹$T$，能够通过模型生成其在隐藏状态空间中的向量表示，即隐藏路径$R$，用如下公式来进行描述：

$$
\mathcal{L} = \max\mathbb{P}(R|T)
$$ (max_post_prob)

在上文的定义中已经提及，$T$是真实世界中的轨迹序列点而$R$是建模过程中假设出来的隐藏状态变量，并不是真实能够观测到的，也很难直接获得$R$分布的解析解。换言之，我们无法通过一个参数已知的数学公式或模型直接计算出给定$T$条件下的$R$的值。

所以在T2VEC中又作如下建模：假设某路径$R$有两条能够观测到的轨迹序列分别是$T_a$和$T_b$（$R$是连续空间中的分布，可以有很多采样的轨迹序列样本）。这两条轨迹序列都是非均匀的（因为真实的轨迹序列是非均匀的），但是$T_a$以低采样率采样而$T_b$是以高采样率采样。因此，我们认为相比于$T_a$，$T_b$更接近$R$，所以可得如下关系:

$$
  \max \mathbb{P}(R|T_a) \approx \max \mathbb{P}(T_b|T_a)
$$ (sim_by_sampling)

注意，因为$R$是隐藏变量，分布无从得知，所以也无法直接从$R$采样获得到$(T_a, T_b)$这样的观测样本组合，我们也没有“标签”来得知两条序列是否由同一个分布产生的。所以在这里，作者使用了如下的巧妙方式来构建了样本组合$(T_a, T_b)$:

* 在真实数据集中随机采样一条轨迹序列$T_{original}$，然后以采样率$r_a$从$T_{original}$随机采样一条序列$T_a$，以采样率$r_b$从$T_{original}$随机采样一条序列$T_b$，其中$r_b > r_a$，此时，就获得了一对来源于同一个路径$R$的观测样本组合$(T_a, T_b)$（基于上文假设的条件下）。

通过这种方式，就解决了计算$\max\mathbb{P}(R|T)$时的标签问题。在seq2seq*编码器-解码器*（encoder-decoder）架构中, 输入是$T_a$，标签是$T_b$，使用编码器将$T_a$编码成高维的向量表示$v$，此时$v$即为上文方法建模中的隐藏路径$R$的向量表示，再通过解码器逐步解码生成$\tilde{T}_b$。

模型训练之后，如果$\tilde{T}_b$和$T_b$足够接近（比如在最小均方误差或交叉熵损失的衡量指标下），那么说明模型很好的捕获了轨迹序列$T$的向量特征$v$。

更近一步的，为了使得模型学习到的特征对数据中的噪声更加的鲁棒，在构建样本的时候针对样本进行了不同程度上的*扭曲*(distortion，在这里表现为添加噪声)，如下:

* 对于序列 $T_a$ ，随机采样其中的一个轨迹点 $P(p_x, p_y)$ ，对其添加高斯扰动:

  $$
  p\prime_x=p_x + 30·d_x
  $$ (distortion_1)

  $$
  p\prime_y = p_y + 30·d_y
  $$ (distortion_1)

其中，$d_x \sim Gaussian(0,1), d_y \sim Gaussian(0,1)$。通过此种样本的构建方式，结合模型的训练，能够很好的应对之前所提到的三个问题: **非均匀分布**、 **低采样率**、 **带噪声**。

```{tip}
在没有标签的情况下，通过对原样本的增强（采样、扭曲、掩码等）操作，生成新的样本并与原样本构成一对样本来训练模型，是如今无监督对比学习的常用样本构建方式，比如计算机视觉领域的[MAE(2021)](https://arxiv.org/pdf/2111.06377)、自然语言处理领域的[BERT(2019)](https://arxiv.org/pdf/1810.04805)。
```


### 优化函数

在{math:numref}`post_prob`中，不难看出，解码器的输出头实际上是一个分类模型，将综合了历史轨迹序列特征和编码器特征的隐藏变量$h_t$经过一个线性层（$W$）和$\mathrm{softmax}$层之后输出词表中每一个位置元素的预测概率。

对于此分类问题的损失函数，在基于seq2seq的自然语言处理任务中一般使用*负对数似然损失*（negative log-likelihood loss）：

$$
  \mathcal{L_1}=-\log \prod_{t}\mathbb{P}(y_t \vert y_{1:t-1}, x)
$$ (nll_loss)

该损失如果直接用于轨迹序列的任务，则会存在如下缺点: 假设在时间点$t$上的标签是$y_t$ ，此外还有另外两个候选结果，$y_1$和$y_2$，并且$\mathrm{distance}(y_t, y_1)>\mathrm{distance}(y_t, y_2)$，此时损失函数对于两者的惩罚是相同权重的，这是不合理的，应该对于距离标签$y_t$更远的候选标签$y_1$施加更多的惩罚，在{math:numref}`nll_loss`中没有体现这种空间近邻关系，而这种空间近邻的先验知识是可以引入到损失函数中的。

因此，T2VEC中提出了基于空间距离加权的损失函数，即*空间近邻感知损失*（spatial proximity aware loss），如下:

$$
  \mathcal{L_2}=-\sum_{t=1}^{|y|}\sum_{u \in V}w_{uy_t} \log \frac{\exp(W_u^Th_t)}{\sum_{v \in V}\exp(W_v^Th_t)}
$$ (spa_loss)

其中:

$$
  w_{uy_t}=\frac{\exp(-||u-y_t||_2/\theta)}{\sum_{v \in V}\exp(-||v - y_t||_2/\theta)}
$$ (spa_weight)

可以看出，其中$u$为词表中的候选标签，$y_t$为真实的标签，$||u-y_t||_2$定义为两个网格中心点的*欧式距离*（euclidean distance）。此时，$w_{uy_t}$候选标签和真实标签的距离经过$\mathrm{softmax}$函数归一化的权重，当$||u-y_t||_2$越大则$w_{uy_t}$越大，损失的惩罚比重更大。

当模型要预测的类别较少时，比如ResNet在ImageNet数据集只有1000个类别，$\mathrm{softmax}$函数的计算是没有问题的，但是当模型要预测的类别非常大时，比如在轨迹序列领域和自然语言处理领域中词表可能是万级别的大小，此时$\mathrm{softmax}$的计算会很慢并且难以训练。

也就是说，{math:numref}`spa_loss`在训练时需要对整个词表做两次求和计算,当字典很大时，计算复杂度很大。

$$
  \sum_{u \in V}w_{uy_t}(w_{u}^Th_t-\log \sum_{v \in V} \exp(W_v^Th_t))
$$ (spa_loss_2)

对于这种问题，对于此类问题，自然语言处理领域通常的解决方法是*噪声对比估计*（noise constrative estimation）。

```{tip}
关于噪声对比估计，可以参见[NCE](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)和[Candidate Sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf)。
```

### 正负样本标签采样

作者基于对真实数据的观察，得出如下两个结论：

* 除了和真实标签$y_t$相邻近的候选标签之外，{math:numref}`spa_weight`中绝大部分$w_{uy_t}$都非常小;
* 对于{math:numref}`spa_loss`，无需精确计算而仅近似计算即可。

基于以上两点，T2VEC中提出了近似计算{math:numref}`spa_loss`{math:numref}`spa_weight`的方法(也及改进的NCE算法):

* 候选标签集合: 计算权重只用计算$K$个距离$y_t$最近的网格即可，定义为$\mathcal{N}_K(y_t)$。
* 负标签集合：使用NCE中的方法来采样负标签集合，即从$V-\mathcal{N}_K(y_t)$中采样负标签集合 $\mathcal{O}(y_t)$，其中$V$表示词表全集。

所以公式{math:numref}`spa_loss`{math:numref}`spa_weight`可以进一步推导为如下:

$$
  \mathcal{L}_3=-\sum_{t=1}^{|y|}\sum_{u \in \mathcal{N}_k(y_t)}w_{uy_t}(W_u^Th_t-\log\sum_{v \in \mathcal{NO}}\exp(W_v^Th_t))
$$ (spa_loss_with_nce)

其中，

$$
  w_{uy_t}=\frac{exp(-||u-y_t||_2/\theta)}{\sum_{v \in \mathcal{N}_K(y_t)}\exp(-||v-y_t||_2/\theta)}
$$ (spa_weigth_with_nce)

$$
  \mathcal{NO}=\mathcal{N}_K(y_t) \cup \mathcal{O}(y_t)
$$ (nce_canidate_set)

```{tip}
对于负标签采样的方法，可参见Tensorflow中的[sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss)和[log_uniform_candidate_sampler](https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/candidate_sampling_ops.py) 的实现。
```


## 代码示例

我们以`Notebook`的形式，使用`TrajDL`来实现T2VEC算法在Porto数据集上的训练和推理，参见[T2VEC实践](../examples/t2vec_example.md)。


```{tip}
1. 本文介绍了轨迹相似度任务、现状和难点。
2. 本文介绍了T2VEC算法，该算法基于seq2seq模型，使用样本对比增强和空间近邻的损失函数进行改进，能够捕获到高质量的序列表征。
```