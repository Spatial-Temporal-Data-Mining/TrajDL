---
jupytext:
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

# Sequence and Dataset

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

```

`TrajDL`主要处理时空序列数据上的深度学习任务，因此在`TrajDL`里面对时空序列数据进行了抽象，分为**位置序列**和**轨迹序列**两种。

这两种序列都是表示一个实体的移动轨迹，只不过前者是位置id组成的序列，后者是轨迹点组成的序列。

`TrajDL`针对上述两种类型的数据，按单条序列和多条序列分别定义了**序列**和**数据集**的概念。
- 前者一般表示单条序列，用户可以将自己的数据集处理成多条这样的序列。
- 对于一个包含多条序列的数据集，我们一般使用后者**数据集**表示。

+++

```{figure} ../_static/images/sequence_and_dataset.svg
:alt: 序列与数据集的关系图
:align: center

序列有一个抽象基类是`BaseSeq`，数据集有一个抽象基类`BaseArrowDataset`。这两项都是按**位置序列**和**轨迹序列**进行细分。

```

```{tip}
未来`TrajDL`会考虑将两种序列的API完全合并，目前大部分接口是相同的。
```

+++

## Sequence

对于单条序列，`TrajDL`提供了两种基础的序列类型用来表示**位置序列**和**轨迹序列**，分别是`LocSeq`和`Trajectory`。

举个例子：我们有`A`、`B`、`C`、`D`是4个不同的位置，一个人接连到达了这四个位置，那么这个人经过的位置序列可以表示为`[A, B, C, D]`。

`TrajDL`提供了一个简单的类型`LocSeq`可以承载这样的位置序列：

```{code-cell} ipython3
:caption: LocSeq Example
:lineno-start: 1

from trajdl.datasets import LocSeq

locseq = LocSeq(["A", "B", "C", "D"])
print(locseq)
```

对于轨迹数据，假设我们有一组经纬度点，表示一个人的实时GPS数据：
```python
[-8.608833, 41.147586]
[-8.608707, 41.147685]
[-8.608473, 41.14773]
[-8.608284, 41.148054]
[-8.607708, 41.148999]
[-8.60742, 41.149719]
```

我们可以使用`Trajectory`类型来表示这个数据

```{code-cell} ipython3
:caption: Trajectory Example
:lineno-start: 1

from trajdl.datasets import Trajectory

traj = Trajectory([
    [-8.608833, 41.147586],
    [-8.608707, 41.147685],
    [-8.608473, 41.14773],
    [-8.608284, 41.148054],
    [-8.607708, 41.148999],
    [-8.60742, 41.149719]
])
print(traj)
```

````{note}
`Trajectory`还可以通过`numpy`创建：

```{code-block} python

import numpy as np
from trajdl.datasets import Trajectory

seq = np.array([
    [-8.608833, 41.147586],
    [-8.608707, 41.147685],
    [-8.608473, 41.14773],
    [-8.608284, 41.148054],
    [-8.607708, 41.148999],
    [-8.60742, 41.149719]
])

traj = Trajectory(seq)
print(traj)
```

效果是一样的。
````

+++

这两种数据类型本质上除了输入的序列不同（一个是位置序列，一个是轨迹点序列），其他的属性和方法都是一致的。可以获取它的长度，它的id等信息

```{code-cell} ipython3
:caption: Sequence Example
:lineno-start: 1

print(len(locseq))
print(len(traj))

print(locseq.entity_id)
print(traj.entity_id)
```

上述两种类型的数据都是单条序列的数据，可以表示一条位置序列、或者一条轨迹序列。它的优点是清晰、简单，主要是给用户提供一个数据管理的工具。

这两种类型的序列数据目前具有6种属性：

| Attribute   | Type                           | Description |
|-------------|--------------------------------|-------------|
| `seq`       | `List[str]` or `numpy.ndarray` | 这一项表示序列的位置信息。`LocSeq`的这一项是`List[str]`，表示位置序列；`Trajectory`的这一项是`numpy.ndarray`，表示经纬度序列，numpy.ndarray一定要是2列，第一列是经度，第二列是纬度。  |
| `entity_id` | `str`                          | id项，这一项用户可以根据自己的需求设定，可以设定为序列的id，或者实体的id。                                                     |
| `ts_seq`    | `List[int]`                    | 这一项表示每个位置的时间戳，可以是到达时的时间戳，也可以是离开时的时间戳，暂时没有区分，时间戳用int类型，具体的单位由用户自行决定。 |
| `ts_delta`  | `List[float]`                  | 表示两个连续的位置之间的时间戳的差值。                                                                                      |
| `dis_delta` | `List[float]`                  | 表示两个连续的位置之间的位移差值。                                                                                         |
| `start_ts`  | `int`                          | 表示这条序列的起始时间戳，具体的单位由用户自行决定。                                                                         |

其中只有第一项是要求用户一定要传入的，其他项都是可选的项，可以直接通过`.entity_id`这样的属性进行获取。

```{note}
目前这样的序列格式还不能满足用户灵活自定义额外的特征，比如一些连续性特征和离散型特征，会在未来版本提供支持。
```

+++

## Dataset

对于多条序列，`TrajDL`提供了基于`Arrow`的数据管理方式，称为`BaseArrowDataset`，细分为`LocSeqDataset`和`TrajectoryDataset`。

这两种数据集是为了管理多条序列使用，它的主要特性是：

高性能
: 基于列存的`Arrow`数据类型提供了高效的存储与查询性能。

零拷贝
: 基于`Arrow`的数据集在取数的时候可以避免数据拷贝，而且基于`Arrow`的数据集在`Pytorch dataloader`处于多进程时的状态下也可以避免数据拷贝。

扩展性好
: 基于`Arrow`的数据集可以与`Polars`、`Pandas`、`Numpy`等成熟工具以极低的损耗无缝转换。用户可以基于`Polars`快速进行数据处理，然后一键导入成`BaseArrowDataset`，而且`BaseArrowDataset`也可以快速转换为`Polars DataFrame`或`Pandas DataFrame`。而且`Arrow`与`Parquet`的交互也非常简单，很自然就可以扩展到分布式计算框架`Spark`上，这让大数据平台处理后的数据集无缝导入到`TrajDL`成为可能。

+++

### LocSeqDataset

`LocSeqDataset`与`LocSeq`是一一对应的，也就是说`LocSeq`具有什么样的属性，那么`LocSeqDataset`就具有什么属性。我们可以通过下面的例子展示如何快速构建一个`LocSeqDataset`。

```{code-cell} ipython3
from trajdl.datasets import LocSeqDataset

seq1 = LocSeq(["A", "B", "C"])
seq2 = LocSeq(["C", "D"])

ds = LocSeqDataset.init_from_loc_seqs([seq1, seq2])
print(ds)
print(len(ds))
```

我们可以通过`to_polars`方法快速将其转换为`Polars DataFrame`进行其他操作。

```{code-cell} ipython3
df = ds.to_polars()
df.head()
```

```{code-cell} ipython3
import polars as pl

# 计算一共有多少个不同的位置
df.select(pl.col("seq").explode().unique()).count().item()
```

由于`LocSeqDataset`的底层使用的是`Arrow`类型的数据结构，因此也是可以直接从`Arrow`层面进行操作的。

```{code-cell} ipython3
ds.seq
```

```{code-cell} ipython3
import pyarrow as pa

# 计算每条序列的长度
pa.compute.list_value_length(ds.seq).to_numpy()
```

我们可以看一下`LocSeqDataset`的schema

```{code-cell} ipython3
ds.schema()
```

对于索引操作，`LocSeqDataset`在设计上是返回一个自身的一个视图，并不会copy一份新的数据出来。而且索引操作的返回结果仍然是一个`LocSeqDataset`类型。

```{note}
为什么索引的返回结果仍然是`LocSeqDataset`？

因为`LocSeqDataset`与`TrajectoryDataset`是构建在`pyArrow.Table`类型上的，`Arrow`数据的列存性质使得`Arrow`数据只有列的概念，没有行的概念，因此按行取数的过程实际是遍历所有的列，根据行的下标进行取数，这样得到的数据如果使用`tuple或者list`表示都不合理，但是是可以使用原来的schema将数据重新组织起来的，并且由于零拷贝的特性，这一点不会有什么损耗。

这样的特性也同样适用于`TrajectoryDataset`，因为这个特性是抽象基类`BaseArrowDataset`提供的。
```

```{code-cell} ipython3
ds[0], ds[1]
```

```{code-cell} ipython3
ds[0].seq, ds[1].seq
```

### TrajectoryDataset

`TrajectoryDataset`和`LocSeqDataset`基本是一样的，唯一的区别在于seq的类型，我们可以创建一个`TrajectoryDataset`试试。

```{code-cell} ipython3
import numpy as np
from trajdl.datasets import Trajectory, TrajectoryDataset

traj1 = Trajectory(seq=np.random.uniform(size=(10, 2)))
traj2 = Trajectory(seq=np.random.uniform(size=(15, 2)))
traj3 = Trajectory(seq=np.random.uniform(size=(5, 2)))

ds = TrajectoryDataset.init_from_trajectories([traj1, traj2, traj3])
print(ds)
```

```{code-cell} ipython3
# 看一下schema
print(ds.schema())
```

可以看到，`seq`的类型与`LocSeqDataset`的`seq`的类型是不同的。

下面是这两种类型的Schema的对比：

| Attribute   | LocSeqDataset                      | TrajectoryDataset                          |
|-------------|------------------------------------|--------------------------------------------|
| `seq`       | `pa.large_list(pa.large_string())` | `pa.large_list(pa.list_(pa.float64(), 2))` |
| `entity_id` | `pa.large_string()`                | `pa.large_string()`                        |
| `ts_seq`    | `pa.large_list(pa.int64())`        | `pa.large_list(pa.int64())`                |
| `ts_delta`  | `pa.large_list(pa.float32())`      | `pa.large_list(pa.float32())`              |
| `dis_delta` | `pa.large_list(pa.float32())`      | `pa.large_list(pa.float32())`              |
| `start_ts`  | `pa.int64()`                       | `pa.int64()`                               |

只有`seq`字段的类型是不同的。

+++

### 数据集的持久化

`BaseArrowDataset`是支持**存储**与**加载**的，通过`save`方法和`init_from_parquet`方法。

调用`save`方法后数据集会以`parquet`的格式存储到磁盘上。

调用`init_from_parquet`之后，数据集会从`parquet`文件重新加载。

```{code-cell} ipython3
help(LocSeqDataset.save)
```

```{code-cell} ipython3
help(LocSeqDataset.init_from_parquet)
```

我们演示一下如何实现数据的存储与加载：

```{code-cell} ipython3
import os
import tempfile

with tempfile.TemporaryDirectory() as tmp_folder:
    path = os.path.join(tmp_folder, "test_ds.parquet")
    print("dataset will be saved as:", path)

    ds.save(path)

    new_ds = TrajectoryDataset.init_from_parquet(path)
    print(new_ds)
```

再数据集处理好后，使用`BaseArrowDataset`存储起来对后续的训练过程很有帮助，如何在训练的时候使用`BaseArrowDataset`需要看后续文档。

+++

```{tip}

本文主要介绍`TrajDL`里面的**序列**与**数据集**的概念。前者用于表示单条序列，后者用于表示多条序列。

`TrajDL`内的大量操作会围绕这些序列和数据集展开，如轨迹切分、数据扰动，Tokenization等。

**数据集**在`TrajDL`的模型训练中发挥很大的作用。

基于`Arrow`的`BaseArrowDataset`还有很多用法会在后续的**高级文档**内展开。

```
