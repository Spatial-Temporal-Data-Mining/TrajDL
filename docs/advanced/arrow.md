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

# BaseArrowDataSet

```{attention}
在1.0.0版本发布前，当前文档的内容可能会发生变化。
```

`BaseArrowDataset`是`TrajDL`在模型训练时主要使用的数据集，它的底层数据全都是用`pyarrow.Table`进行存储的，这就意味着`BaseArrowDataset`与任何支持`Arrow`的工具都是可以打通的。在`Python`科学计算领域里与`Arrow`关系紧密的框架有`Polars`，`Pandas`等，当然还有`Spark`。这些框架可以利用**支持`Arrow`的特性**直接与`BaseArrowDataset`进行交互。

+++

## init_from_table

`BaseArrowTable`有一个`init_from_table`的方法，支持直接从`polars.DataFrame`，`pandas.DataFrame`，`pyarrow.Table`加载数据到`BaseArrowTable`内。这个方法要求用户清楚的知道`BaseArrowTable`的schema。

我们以`LocSeqDataset`为例进行讲解，我们首先可以通过`.schema()`方法获取其`pyarrow.Table`的schema。

```{code-cell} ipython3
from trajdl.datasets import LocSeqDataset
```

```{code-cell} ipython3
LocSeqDataset.schema()
```

可以看到，一共有六列，并且每一列都有已经设置好的类型，那么如果我们拿到一个符合这个schema的`pyarrow.Table`，就可以直接构建一个`LocSeqDataset`，我们用`Polars`做一个演示。

```{code-cell} ipython3
import polars as pl

# 构建一个只有2列的DataFrame
df = pl.DataFrame({
    "seq": [
        ["a", "b", "c"],
        ["b"],
        ["c", "d"],
    ],
    "entity_id": ["1", "2", "3"],
})
df.head()
```

```{code-cell} ipython3
# 转换成pyarrow.Table看一下
df.to_arrow()
```

```{code-cell} ipython3
# 通过init_from_arrow可以直接加载
ds = LocSeqDataset.init_from_arrow(df.to_arrow())
ds
```

```{code-cell} ipython3
ds.schema()
```

```{code-cell} ipython3
ds.seq
```

```{code-cell} ipython3
ds.entity_id
```

可以看到，我们通过`Polars`构建了一个简单的`polars.DataFrame`，里面只有3条序列，并且只有2个属性，这两个属性与`LocSeqDataset`定义的一致，`Polars`底层的`Arrow Table`就可以快速导入到`LocSeqDataset`里面，因为使用的是`Arrow`，底层没有发生数据拷贝，因此这个转换的性能是极高的。

同理，这种转换也支持`Pandas`，`PyArrow`等其他工具。

```{code-cell} ipython3
df.to_pandas()
```

```{code-cell} ipython3
import pyarrow as pa

# 从pandas加载
arrow_table = pa.Table.from_pandas(df.to_pandas())
LocSeqDataset.init_from_arrow(arrow_table)
```

当然，为了方便使用，`TrajDL`将上述的`DataFrame`进行了统一封装，得到了`init_from_table`方法

```{code-cell} ipython3
help(LocSeqDataset.init_from_table)
```

```{code-cell} ipython3
LocSeqDataset.init_from_table(df)
```

```{code-cell} ipython3
LocSeqDataset.init_from_table(df.to_pandas())
```

```{code-cell} ipython3
LocSeqDataset.init_from_table(df.to_arrow())
```

```{tip}
`TrajDL`建议用户使用`Polars`框架，因为`Polars`框架在单机上的性能比较好，其数据类型多样，而且与`Arrow`的交互非常清晰。
```

+++

接下来我们已`TrajectoryDataset`为例来实验一下。

```{code-cell} ipython3
from trajdl.datasets.open_source.conf import PortoDataset

# 只取前两条数据作为演示
df = PortoDataset().load().head(2)
df.head()
```

```{code-cell} ipython3
from trajdl.datasets import TrajectoryDataset

TrajectoryDataset.schema()
```

```{code-cell} ipython3
import polars as pl

# 取出轨迹，TAXI_ID作为entity_id
new_df = df.select(pl.col("POLYLINE").alias("seq"), pl.col("TAXI_ID").cast(pl.String).alias("entity_id"))
new_df.head()
```

```{code-cell} ipython3
ds = TrajectoryDataset.init_from_table(new_df)
ds
```

```{code-cell} ipython3
ds.seq
```

```{code-cell} ipython3
ds.entity_id
```

```{tip}

本文讲解了`BaseArrowDataset`是如何与支持`Arrow`的框架进行交互的，这种交互方式跨过了单条序列`BaseSeq`，直接从科学计算框架的底层数据进行加载，避免了数据拷贝，极大了优化了数据集的构建速度。并且基于`Arrow`的数据集在`Pytorch`框架使用多进程`DataLoader`的时候可以显著减少内存使用。

```

```{code-cell} ipython3

```
