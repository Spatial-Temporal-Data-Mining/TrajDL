---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tokenizer

```{attention}
在1.0.0版本发布前，当前文档的内容可能会发生变化。
```

`TrajDL`在处理序列数据的时候，提供了`Tokenizer`这样的工具。`Tokenizer`可以将位置序列`LocSeq`或轨迹序列数据`Trajectory`转换为一个`List[int]`。因为对于大量轨迹挖掘的模型来说，模型的下层是一个嵌入层（`EmbeddingLayer`），嵌入层需要的是离散的输入，而且在`Pytorch`里面需要提前将其处理成从`0`开始的整数下标，`Tokenizer`就提供了这样的能力，它会管理一个`vocabulary`用来记录每个**位置**对应的整数下标。

+++

## 位置序列

下面我们举例来说明：对于位置序列`LocSeq`，`Tokenizer`如何将**位置**映射为位置的**下标**。

```{code-cell} ipython3
from tqdm.notebook import tqdm

from trajdl.datasets import LocSeq
from trajdl.tokenizers import LocSeqTokenizer

# 构建几个简答的位置序列
locseqs = [
    LocSeq(["A", "B", "C", "D"]),
    LocSeq(["A", "D", "E"])
]

# 构建tokenizer
tokenizer = LocSeqTokenizer.build(locseqs)
```

`Tokenizer`在实现的时候参考了NLP领域的tokenizer，其内部也是通过词库`vocabulary`构建的。可以通过`len`函数获取vocab的大小

```{code-cell} ipython3
# 获取vocab的大小
len(tokenizer)
```

```{note}

为什么上面的tokenizer vocab的大小是10？

因为`Tokenizer`在构建的时候会将一些**SPECIAL TOKENS**放入vocab里面，比如`[BOS]`, `[EOS]`等。

可以通过下面的属性直接获取：
- `.bos`：起始token，部分场景会将这个token加到序列的起始位置
- `.eos`：结束token，部分场景会将这个token加到序列的结束位置
- `.unk`：未知，已经构建好的Tokenizer在对没有见过的位置进行转换时使用的token
- `.pad`：pad，在对序列进行padding的时候使用
- `.mask`：mask，在进行mask的时候使用

```

```{code-cell} ipython3
# 获取[BOS]的下标
tokenizer.bos
```

```{code-cell} ipython3
# [PAD]的下标
tokenizer.pad
```

除此以外，我们还可以使用`loc2idx`方法直接获取某个位置对应的下标。

```{code-cell} ipython3
# 获取A的下标
tokenizer.loc2idx("A")
```

```{code-cell} ipython3
# 获取F的下标，我们在上面构建的位置序列里面是没有这个位置的，因此它应该返回[UNK]的下标
tokenizer.loc2idx("F")
```

```{code-cell} ipython3
assert tokenizer.loc2idx("F") == tokenizer.unk
```

可以看到，对于已经构建好的`Tokenizer`，如果让它去转换一个没有见过的位置，它会用`[UNK]`代替。

+++

下面是`Tokenizer`对位置序列进行编码的例子：

```{code-cell} ipython3
# 对第一条位置序列编码
tokenizer.tokenize_loc_seq(locseqs[0])
```

```{code-cell} ipython3
# 对第二条序列编码，并且在首尾分别增加BOS与EOS的token，返回的序列用numpy.ndarray的类型
tokenizer.tokenize_loc_seq(locseqs[1], add_bos=True, add_eos=True, return_as="np")
```

```{code-cell} ipython3
# 对第二条序列编码，不要BOS，要EOS，用torch.LongTensor的类型返回
tokenizer.tokenize_loc_seq(locseqs[1], add_bos=False, add_eos=True, return_as="pt")
```

可以看到，`Tokenizer`是支持三种类型的返回值的：`List[int]`、`numpy.ndarray`、`torch.LongTensor`。并且支持是否添加`[BOS]`或`[EOS]`标签。

+++

## 轨迹序列

位置序列已经是由位置组成的序列了，而轨迹序列其原始数据是一组轨迹点，因此轨迹序列如果也希望转换成`List[int]`，需要引入网格系统`GridSystem`。网格系统可以将经纬度点转换成网格id，以此表示位置。这样位置序列与轨迹序列就都可以表示成位置序列了，然后通过`Tokenizer`转换为模型的输入，即`List[int]`。

```{note}

那么对于所有轨迹数据，我只要提前定义好网格系统，然后全都转换成位置序列不就好了？也就是我将`TrajectoryDataset`预先配合网格系统`GridSystem`转换为`LocSeqDataset`。

这种方法在部分场景是没有问题的，但是有些场景需要随着训练的过程对原始轨迹数据进行扰动，也就是直接对原始的经纬度进行操作，如果预先做了转换，这一步的扰动就比较难做。因此`TrajDL`是专门为轨迹数据设计了`Tokenizer`，这样可以保证训练的时候读取的是原始轨迹数据，可以随时操作原始轨迹数据，然后再通过`Tokenizer`将轨迹数据转换为`List[int]`。
```

对于轨迹数据，我们要利用网格系统将其经纬度转换为位置。我们先加载几条轨迹数据。

```{code-cell} ipython3
from trajdl.datasets import Trajectory
from trajdl.tokenizers import T2VECTokenizer

from trajdl.datasets.open_source.conf import PortoDataset
```

```{code-cell} ipython3
# 加载Porto数据集，取出轨迹这一列
polylines = PortoDataset().load().head()[["POLYLINE"]]
print(polylines)
```

```{code-cell} ipython3
# 将polyline转换为`Trajectory`
trajs = [
    Trajectory(seq=polyline.to_numpy(), entity_id=str(idx))
    for idx, polyline in enumerate(polylines["POLYLINE"])
]
```

```{code-cell} ipython3
trajs
```

目前`TrajDL`提供了`T2VECTokenizer`，在**T2VEC**里面，作者使用原始坐标(WGS84坐标系)定义了一个区域，只有在这个区域内的坐标会在实验中使用。但是将坐标转换为位置的时候，是将所有经纬度转换到了Web Mercator坐标系下。因此`TrajDL`封装的`T2VECTokenizer`在构造的时候需要用户指定这两项，即一个基于WGS84坐标系的`Boundary`，还有一个基于Web Mercator坐标系的`GridSystem`。

```{code-cell} ipython3
import polars as pl

# 获取经纬度的最大、最小值
lng = polylines.explode("POLYLINE").select(pl.col("POLYLINE").arr.first())
min_lng, max_lng = lng.min().item(), lng.max().item()

lat = polylines.explode("POLYLINE").select(pl.col("POLYLINE").arr.last())
min_lat, max_lat = lat.min().item(), lat.max().item()
print(min_lng, max_lng, min_lat, max_lat)
```

```{code-cell} ipython3
from trajdl.trajdl_cpp import RectangleBoundary
from trajdl.grid.base import SimpleGridSystem

# 使用经纬度的最大最小值构建Boundary
wgs_boundary = RectangleBoundary(
    min_x=min_lng,
    min_y=min_lat,
    max_x=max_lng,
    max_y=max_lat
)

# 将Boundary转换为Web Mercator坐标系后再构建网格系统，这样网格的划分就是在Web Mercator坐标系了
grid = SimpleGridSystem(boundary=wgs_boundary.to_web_mercator(), step_x=100.0, step_y=100.0)

# 之所以还要传入原始坐标系的Boundary，那是因为在T2VEC的算法中还是希望用原始坐标系来圈定数据的使用范围
tokenizer = T2VECTokenizer.build(
    grid=grid,
    boundary=wgs_boundary,
    trajectories=trajs,
    max_vocab_size=10000,
    min_freq=1
)

# 打印网格的个数和tokenizer里面vocab的个数
print(len(grid), len(tokenizer))
```

可以看到，网格的大小是7680，tokenizer vocab的大小是136，后者的136里面还包含了**SPECIAL TOKENS**，比如`[BOS]`，`[EOS]`等。

```{code-cell} ipython3
print(trajs[0])

# 对第一条轨迹数据进行编码
tokenizer.tokenize_traj(trajs[0])
```

对于`T2VECTokenizer`来说，如果连续的两个经纬度坐标同属于同一个网格，那么这两个经纬度变换得到的网格最终只会保留1个，所以`T2VECTokenizer`编码后的序列不会出现连续两个相同的位置，因此其长度也会小于等于原始轨迹序列的长度。

+++

```{tip}

本文介绍了`Tokenizer`，`Tokenizer`用于将**位置序列**或者**轨迹序列**进行编码，转换为`List[int]`作为模型的输入样本，并且像`LocSeqTokenizer`在实现的时候也考虑了针对**位置**的出现频率进行倒序排列，以此支持类似`SampledSoftmax`这样的损失函数。而且`Tokenizer`还提供了对**SPECIAL TOKENS**的管理，用户无需担心手动管理这些特殊的tokens。

```

```{code-cell} ipython3

```
