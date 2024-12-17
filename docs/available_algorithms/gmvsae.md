# GM-VSAE

[[ICDE2020] Online anomalous trajectory detection with deep generative sequence modeling](https://kaiqizhao.github.io/ICDE20_CR.pdf)


## 摘要

**轨迹序列异常检测**是轨迹序列领域中的重要任务之一，在诸多方面都有应用的场景。比如在城市中从$A$地点打车到$B$地点，一般而言有几条被大量司机驶行的固定路径，如果某用户打车时司机的轨迹并非与这些固定路径相似，那么极有可能是被绕路了，也即轨迹序列异常。

在此，作者提出一种轨迹序列异常计算算法——**GM-VSAE**（Gaussian Mixture Variational Sequence AutoEncoder），相比于前人的方法，作者认为GM-VSAE存在以下优点:

* 能够捕获复杂轨迹序列的表征。
* 在隐藏状态空间中使用高斯混合分布表征不同类型的路径。
* 基于轨迹生成的在线检测（detection-via-generation）。

## 预备知识

轨迹序列定义为按时间顺序采集的GPS坐标点序列 $P=\{p_1 \to p_2 \to \ldots \to p_n\}$。路径表示轨迹序列在物理世界中所遵循的连续路线，一条轨迹序列可以被视为路径的一条采样样本。

通常，在某个**固定行程**（即某两位置之间的行程）中的一条*轨迹*（trajectory），如果没有遵循*正常路径*（normal route）的话，则可以被认为是**异常轨迹**。注意，在某固定形成之间亦可以存在多条正常路径。下图中 ${S_1, S_2}$表示行程的起点， ${D_1, D_2}$表示行程的终点，其中 ${T_1, T_2}$ 两条轨迹序列被视为是*异常轨迹*（anomalous trajectory）。

```{figure} ../_static/images/gmvsae_detour_example.jpg
:alt: 异常轨迹示例
:width: 600px
:align: center
:name: anomaly-trajectory

图1. 异常轨迹示例（图源：[GM-VSAE](https://kaiqizhao.github.io/ICDE20_CR.pdf)）
```


## GM-VSAE


### 定义

为了后续的GM-VSAE的建模，作者在此作如下定义：

* $r$：路径，定义$p(r)$为一条路径被轨迹行驶的概率。更大的$p(r)$意味着$r$更有可能是一条正常路径。
* $r_*$：正常路径，定义$p(T|r_*)$表示轨迹$T$由正常路径$r_*$生成的概率。更小的$p(T|r_*)$意味着$T$更有可能是一条异常轨迹。
* **在线异常序列检测**：给定一条序列 $T$ ，其起点是 $S_T$，终点是 $D_T$ ，在线异常序列检测任务是：（1）挖掘 $S_T$和$D_T$之间的正常路径；（2）计算当前轨迹由正常路径生成的概率$p(T|r_*)$，通过该值来在线判定是否是异常轨迹。


为了实现上文中的两个目的，受[VAE](https://arxiv.org/pdf/1312.6114)启发，作者提出了**GM-VSAE**算法，即高斯混合变分序列自编码器，该算法可以实现如下三个功能：

* 以向量来表征路径在隐空间中的分布。
* 建模路径的分布 $p(r)$。
* 给定路径，生成该路径下的一条轨迹序列（采样$p(T|r)$）。


### 方法建模

GM-VASE原理架构图如下:

```{figure} ../_static/images/gmvsae_architecture.jpg
:alt: GM-VSAE整体架构
:width: 600px
:align: center
:name: gm-vsae-arch

图2. GM-VSAE整体架构（图源：[GM-VSAE](https://kaiqizhao.github.io/ICDE20_CR.pdf)）
```
#### 嵌入表征层

原始的轨迹序列无法直接作为模型的输入，需要先对地图构建网格系统将轨迹转换为离散的词元序列，然后可使用词嵌入模型来构建嵌入表征层，将词元序列中的每一个词元转换为*嵌入向量*（embedding）作为GM-VSAE后续模块的输入。具体细节可以参见[Word2Vec](https://arxiv.org/pdf/1301.3781)，此处不做赘述。

#### 编码器

[图2](#gm-vsae-arch)中Inference Network即为编码器，定义为$q_\phi(r \vert T)$，即在给定轨迹序列$T=(t_1, t_2, \ldots, t_n)$的情况下，计算其在隐空间中的向量表示$r_T \in \mathbb{R}^M$。

在GM-VSAE中编码器使用的是RNN系列的模型。具体地，在第 $i$ 步时，编码器输出隐藏状态（hidden state）：

$$
h_i = f_1(t_i, h_{i-1}) \quad  i=1,2,3,\ldots,n,
$$ (encoder-hidden_state)

其中，$h_i \in \mathbb{R}^K$，$K$即为隐藏层的维度。

参照[VAE](https://arxiv.org/pdf/1312.6114)中对隐藏状态的建模，GM-VSAE将$q_\phi(r \vert T)$建模为高斯分布，如下：

$$
r_T \sim q_\phi(r \vert T)=\mathcal{N}(\mathbf{\mu_T}, \mathbf{\sigma}_T^2\mathbf{I})
$$ (gaussian_post_prob)

其中，$\mu_T \in \mathbb{R}^M$，$\sigma_T \in \mathbb{R}^M$，${\mu_T, \sigma_T}=g_1(\mathbf{h}_n)$，$g_1$即为编码器的模型部分。

所以，我们将轨迹序列$T$作为编码器的输入，此时编码器的$g_1$输出的均值$\mu_T$和标准差$\sigma_T$即为轨迹$T$在隐藏状态空间中的向量表征。


```{tip}
1. 此处编码器的建模和下文隐藏状态空间的建模，其基本原理在[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)一文中有更为详细的讲解。
2. 在[T2VEC](./t2vec.md)中亦有针对轨迹领域的建模方式讲解。
```

#### 隐藏状态空间

通过编码器可以得到轨迹$T$在隐空间中的向量表示，但是我们依旧无法区分在隐藏状态空间中哪些路径是正常的。因此，作者使用高斯混合分布来对隐藏路径进行建模。

作者假设在隐藏状态空间中存在$C$中不同类型的路径${r}$分布，其中不同类型的路径在物理世界中有不同的含义，比如道路类型、旅行区域等等。为了建模不同类型的隐藏路径分布，作者定义了如下两个概率分布：

* $p_\gamma(c)=Mult(\pi)$，其中$\pi \in \mathbb{R}^C$。$p_\gamma(c)$即为不同类型的隐藏路径在隐藏状态空间中的概率分布。
* $p_\gamma(\mathbf{r}|c) = \mathcal{N}(\mu_c, \sigma_c^2\mathbf{I})$ ，其中$\mu_c \in \mathbb{R}^M$，$\sigma_c \in \mathbb{R}^M$分别表示该高斯分布的均值和标准差。$p_\gamma(\mathbf{r}|c)$用以建模在路径类型$c$中路径下$\mathbb{r}$出现的概率。

可见，上述两个概率分布共同来对隐藏状态空间建模，即$p_\gamma(\mathbf{r},c)=p_\gamma(\mathbf{r} \vert c)p_\gamma(\mathbf{c})$，隐藏状态空间模型的参数是$\gamma=\{\mathbf{\pi}, \mathbf{\mu_c}, \mathbf{\sigma_c}\}$。

由于新增了隐藏路径类型$c$这个变量，所以可以将编码器推理路径$\mathbf{r}$扩展为推理路径$\mathbf{r}$和路径类型$c$，即$q_\phi(\mathbf{r}, c|T)$。借助*平均场近似*（Mean-field approximation），我们可以将其分解为如下：

$$
q_\phi(\mathbf{r}, c|T) = q_\phi(\mathbf{r}|T)q_\phi(c|T)
$$ (mean-field-decomp)

其中$q_\phi(c|T)$可以近似为如下：

$$
q_\phi(c|T):=p_\gamma(c|\mathbf{r}_T)=\frac{p_\gamma(c)p_\gamma(\mathbf{r}_T|c)}{\sum_{i=1}^Cp_\gamma(c_i)p_\gamma(\mathbf{r}_T|c_i)}
$$ (post_sim_1)

其中，$r_T$采样于分布$q_\phi(\mathbf{r}|T)$。

#### 解码器

解码器的任务是根轨迹序列$T$在潜在空间中的向量表示，来逐步生成轨迹$T'$。为此，作者进行了如下建模，对于第$i$步的序列点$t_i$的生成，其与下面两个因素相关:

* 隐藏路径$\mathbf{r}$
* 先前的序列$t_{<i}=t_1 \to t_2 \to \ldots \to t_{i-1}$

所以我们有：

$$
t_i \sim p_\theta(t_i|t_{<i}, r)
$$ (decoder-model)

在解码器逐步解码的范式下，将先前的序列$t_{<i}$和隐藏路径$\mathbf{r}$编码成隐藏状态向量（hidden state vector）：

$$
\mathbf{g}_i=f_2(t_i, \mathbf{g}_{i-1}) \quad i=1,2,\ldots,n \quad \mathbf{g}_0 = r
$$ (decode-g)

其中$g_i \in \mathbb{R}^k$，$f_2$即为编码器的模型部分，在GM-VSAE中为RNN系列的模型。上述的公式可以进一步写成如下：

$$
t_i \sim p_\theta(t_i|t_{<i}, r)=p_\theta(t|g_{i-1})=Mult(\mathrm{softmax}(g_2(\mathbf{g}_{i-1})))
$$ (decoder-t)

```{note}

多项式分布(Multinomial distribution)用于描述一次实验中从多个类别中选择的情况。在每个类别上都有相应的概率。具体来说，如果你有$K$个类别，每个类别$i$的选择的概率为$p_i$，且满足$\sum_{i=1}^Kp_i=1$，那么从这些类别中采样就是在多项式分布的框架下进行的。

```


### 目标函数

在GM-VSAE中，有如下参数需要优化，分别是嵌入表征层，编码器$\phi=\{f_1(\cdot),g_1(\cdot)\}$，潜在路径模型$\gamma=\{\mathbf{\pi},\mathbf{\mu}_c,\mathbf{\sigma}_c\}$以及解码器$\theta=\{f_2(\cdot),g_2(\cdot)\}$。

参照[VAE](https://arxiv.org/pdf/1312.6114)，在GM-VSAE中目标函数是轨迹序列的最大似然函数：

$$
\log p_\theta(T^{(1)}, T^{(2)}, \ldots,T^{(N)})=\sum_{j=1}^N\mathrm{log}p_\theta(T^{(j)})
$$ (max_log_likelihood)

在这里定义*边际似然下界*（ELBO）为$\mathcal{L}(\theta,\gamma,\phi;T)$，根据VAE中相关的公式的推导，我们可以得到如下公式:

$$
\log p_\theta(T^{(1)}, T^{(2)}, \ldots, T^{(N)}) & = \mathbb{E}_{q_\phi}(\mathbf{r}, c | T) \left[ \log \frac{p_\theta \text{,} \gamma(T, r, c)}{q_\phi(r, c | T)} \right] \\
    & = \mathbb{E}_{{q_\phi}(\mathbf{r}|T)}\left[\log p_\theta(T|\mathbf{r})\right] - \mathbb{E}_{{q_\phi}(\mathbf{r}|T)}\left[D_{KL}(q_\phi(c|T) \Vert p_\gamma(c)\right] \\
    & \quad - \mathbb{E}_{q_\phi(c|T)}\left[D_{KL}(q_\phi(\mathbf{r}|T) \Vert p_\gamma(\mathbf{r}|c))\right]
$$ (elbo)

等式右边第一项表示重构损失，它衡量模型在重构数据的准确性。第二项对应于均匀先验损失，它鼓励混合分布中所有成分的均衡贡献。第三项则是高斯损失，它评估潜在变量与模型中假设的高斯分布之间的吻合程度。

首先$D_{KL}(q_\phi(\mathbf{r}|T) \Vert p_\gamma(\mathbf{r}|c))$衡量两个高斯分布之间的Kullback-Leibler散度，可以表示如下

$$
D_{KL}(q_\phi(\mathbf{r}|T) \Vert p_\gamma(\mathbf{r}|c)) = \frac{1}{2}(\log \sigma_c^2 - \log \sigma_T^2 + \frac{\sigma_T^2}{\sigma_c^2} + \frac{(\mu_T - \mu_c)^2}{\sigma_c^2} - 1)
$$ (kl-div)

因此，我们可以对后验分布$q_\phi(c|T)$计算KL散度的期望，如下所示：

$$
\mathbb{E}_{q_\phi(c|T)}\left[D_{KL}(q_\phi(\mathbf{r}|T) \Vert p_\gamma(\mathbf{r}|c))\right] & = \mathbb{E}_{q_\phi(c|T)} \left[\frac{1}{2}(\log \sigma_c^2 - \log \sigma_T^2 + \frac{\sigma_T^2}{\sigma_c^2} + \frac{(\mu_T - \mu_c)^2}{\sigma_c^2} - 1) \right] \\ & = \frac{1}{2} \sum_{c=1}^{C} q_\phi (c|T)\left[\log \sigma_c^2 - \log \sigma_T^2 + \frac{\sigma_T^2}{\sigma_c^2} + \frac{(\mu_T - \mu_c)^2}{\sigma_c^2} - 1)\right]
$$ (mean_post_1)

我们将$q_\phi (c|T)$定义为$\mathbf{p}_{post}$，计算方式如下：

$$
\mathbb{E}_{q_\phi(c|T)}\left[D_{KL}(q_\phi(\mathbf{r}|T) \Vert p_\gamma(\mathbf{r}|c))\right] & = \frac{1}{2} \sum_{c=1}^{C} q_\phi (c|T)\left[\log \sigma_c^2 - \log \sigma_T^2 + \frac{\sigma_T^2}{\sigma_c^2} + \frac{(\mu_T - \mu_c)^2}{\sigma_c^2} - 1)\right] \\ & = \frac{1}{2} \sum_{c=1}^{C} p_{c,post}\left[\log \sigma_c^2 - \log \sigma_T^2 + \frac{\sigma_T^2}{\sigma_c^2} + \frac{(\mu_T - \mu_c)^2}{\sigma_c^2} - 1)\right] \\ & = \frac{1}{2} \sum_{c=1}^{C} p_{c,post}\left[\log \sigma_c^2 + \frac{\sigma_T^2}{\sigma_c^2} + \frac{(\mu_T - \mu_c)^2}{\sigma_c^2})\right] - \frac{1}{2}(1 + \log \sigma_T^2)
$$ (mean_post_2)

接下来，我们计算第二项，称为均匀损失，它可以重写如下：

$$
\mathbb{E}_{{q_\phi}(\mathbf{r}|T)}\left[D_{KL}(q_\phi(c|T) \Vert p_\gamma(c)\right] & =\int q_\phi(\mathbf{r}|T)\left[D_{KL}(q_\phi(c|T) \Vert p_\gamma(c)\right]d\mathbf{r} \\ & = D_{KL}(q_\phi(c|T) \Vert p_\gamma(c) \\ & = \sum_{c=1}^{C} {p_{c,post}} \log {p_{c,post}} - \log \frac{1}{C} \sum_{c=1}^{C} {p_{c,post}} \\ & = \sum_{c=1}^{C} {p_{c,post}} \log {p_{c,post}} + \log C
$$ (uniform_loss)


#### 在线轨迹序列异常检测

基于上述的编码器-隐藏路径空间-解码器的GM-VSAE模型，训练完成之后，如何进行轨迹序列的异常检测呢？

我们知道，[VAE](https://arxiv.org/pdf/1312.6114)对隐藏状态空间建模，给定输入$x$可以由编码器生成隐藏状态向量$z$，然后$z$经过解码器又可以重构出一个和输入$x$相似的$x'$，我们说这是因为$x$对应的隐藏状态向量$z$服从隐藏状态空间$p(z)$分布，所以从$p(z)$中采样又能恢复出$x'$来。

同理，如果一个轨迹序列$T$是正常的，那么其经过编码器编码而成的隐藏路径$r$必定服从隐藏状态空间分布$p_\gamma(\mathbf{r},c)$，因此当解码器解码$r$时也能够恢复出一条与$T$相似的序列$T'$。如此，计算$T$和$T'$的相似度即可判定该序列是否异常。

但是，上述判定方式仅适合用于离线轨迹序列异常检测，在离线的情况下，轨迹序列$T$已经采集完毕。但是在在线的情况下，轨迹序列是实时生成的，比如在出租车行驶的过程中采集实时的轨迹序列数据，此时如何进行异常检测呢？

在GM-VSAE中，对隐藏状态空间建模为$p(\mathbf{r},c)$，其含义即为：一共有$C$个类型的隐藏路径，每类隐藏路径$r$都服从高斯分布$p_\gamma(r|c)$，均值为$\mu_c$，标准差为$\sigma_c$。所以我们可以使用其均值$\mu_c$来作为$\mathbf{r}$的近似替代。在线异常检测时，无需使用编码器来获得$r$，而是使用$\mu_c$来作为$r$，那么我们只需要用$C$个$\mu_c$，通过解码器计算当前的序列的概率即可实时判定该序列是否为异常：

$$
s(T)=1 -  \underset{c}{\operatorname{argmax}} \exp[\frac{\log p_\theta(T|\mu_c)}{n}]
$$ (online_detection_1)

进一步的，可写出如下更为详细的计算公式：

$$
s(t_{\leq i+1})=1 -  \underset{c}{\operatorname{argmax}} \exp[\frac{\log p_\theta(t_{\leq i}|\mu_c)p_\theta(t_{i+1}|t_{\leq i},\mu_c)}{i+1}]
$$ (online_detection_1)

## 代码示例

我们以`Notebook`的形式，使用`TrajDL`实现了GM-VSAE算法的训练和推理，参见[GM-VSAE代码实践](../examples/gmvsae_example.md)。


```{tip}
1. 本文介绍了轨迹序列异常检测任务。
2. 本文介绍了GM-VSAE的原理部分，针对论文中的公式作近一步更为详细的推导。
```