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

# Collate Function

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

```

因为`TrajDL`是基于`Pytorch Lightning`构建的，本质依赖`Pytorch`的基础工具。因此在模型训练的时候，需要用户定义`Dataset`和`DataLoader`，有些时候还需要定义`Sampler`。我们在前面章节里面讲述的**批数据**，也就是`BaseArrowDataset`，这个就是`Dataset`的子类。用户构建好后需要了解的是如何在`DataLoader`里面操作它。

`TrajDL`当前提供的`BaseArrowDataset`都是[May-style Dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets)。这种`Dataset`可以直接通过下标进行数据的读取。因此这种数据集一般也不会太大，太大会导致内存放不下。

```{tip}
目前`TrajDL`支持的公开数据集都可以使用`Map-style Dataset`实现，因此未来`TrajDL`会尝试支持[Iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets)，这在数据集非常大的场景非常必要。
```

接下来我们说一下`DataLoader`与`Dataset`的关系：
1. 在不指定`Sampler`的情况下，`DataLoader`在生成一个batch的时候会先生成一组index，然后用这一组index去`Dataset`里面获取数据。举个例子，当我们将`batch_size`设定为3的时候，且`shuffle`设定为`False`的时候，第一个batch的index是`[0, 1, 2]`，所以`DataLoader`内部会使用这些下标去`Dataset`里面取数。然后如果`DataLoader`在发现取出来的数据恰好可以组成一个矩阵的时候，`DataLoader`会把它们拼接在一起，形成一个batch的`torch.Tensor`。
2. 但是在`TrajDL`的场景里面，我们一般加载的是原始的**位置序列**或者**轨迹序列**，这些序列的长度并不相同，而且我们可能还需要做一些额外的操作，比如裁剪或扰动，这时候`DataLoader`里面有一个叫做`collate_function`的概念就派上了用场。`collate_function`会在使用下标`[0, 1, 2]`从`Dataset`取数之后发生调用，然后在最终的batch生成之前结束调用。而`collate_function`的返回值就是用户通过`DataLoader`拿到的batch数据。

基于上述信息，我们可以发现，`TrajDL`已经实现好了`Dataset`，而`DataLoader`里面最核心的工作就是`collate_function`，因此很多时候使用`TrajDL`进行新的算法开发的时候，一个重要的工作就是如何从`BaseArrowDataset`里面生成batch数据，即如何编写`collate_function`。

```{note}

对于**Map-style Dataset**，`Pytorch`是如何通过给定下标`[0, 1, 2]`然后进行数据加载的呢？

参照官方文档：[torch.utils.data.Dataset](https://pytorch.org/docs/2.5/data.html#torch.utils.data.Dataset)，文档提到所有`Dataset`的子类需要实现`__getitem__`方法，这个方法是`Python`自带的魔术方法，实现后就可以使用`[idx]`进行数据索引。然而文档内还提到一个`__getitems__`方法，这个并不是`Python`自身的魔术方法，而是`Pytorch`定义的一个方法，而且并不是必须要实现，这个方法的作用是**加速批量数据的加载**。

我们可以找到`DataLoader`的源码，查看`DataLoader`是如何针对给定的indices进行数据加载的：[fetch.py](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/utils/data/_utils/fetch.py#L46)，从源码里面可以看到，`Pytorch`会优先使用`Dataset`的`__getitems__`方法进行数据加载，其次才是使用`__getitem__`方法，取完之后的数据调用了`collate_fn`方法。

这里有一个细节：
- 如果使用的是`__getitems__`，那么传给`collate_fn`方法的参数是`self.dataset.__getitems__(possibly_batched_index)`
- 如果使用的是`__getitem__`，那么传给`collate_fn`方法的参数是`[self.dataset[idx] for idx in possibly_batched_index]`

从`TrajDL`前面的文档可以知道，`BaseArrowDataset`在通过`[idx]`取值的时候，返回的仍然是`BaseArrowDataset`，即`__getitem__`方法的返回值类型是`BaseArrowDataset`。同时，`BaseArrowDataset`也实现了`__getitems__`方法，其返回值类型仍然是`BaseArrowDataset`。这两者的区别在于size不同。因此对于`TrajDL`提供的`BaseArrowDataset`，用户在使用`DataLoader`加载的时候会默认使用`__getitems__`方法，所以传入`collate_fn`的参数一定是一个`BaseArrowDataset`，而不是`List[BaseArrowDataset]`。

```

+++

接下来我们以`TULER`算法为例，演示如何编写一个`DataLoader`的`collate_function`。

`TULER`模型在训练的时候，需要三项内容：
1. 一个位置序列组成的batch，类型是`torch.LongTensor`，shape是`(batch_size, num_timesteps)`
2. 一个`List[int]`，表示每条序列的长度，其size是`batch_size`
3. 一个标签的batch，类型是`torch.LongTensor`，shape是`(batch_size,)`，每一项对应每条序列的实际标签。

这实际上是一个三元组。

那么我们要定义的`collate_function`就很简单了，只要**接受一个batch的输入，返回一个三元组**即可。

由于`BaseArrowDataset`实现了`__getitems__`方法，且返回一个`BaseArrowDataset`类型，这个batch的输入就是一个`BaseArrowDataset`，其size等于`batch_size`。

```{code-cell} ipython3
import polars as pl
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from trajdl.datasets.open_source import GowallaDataset
from trajdl.datasets import LocSeq, LocSeqDataset
from trajdl.tokenizers import LocSeqTokenizer

# 取id小于5的用户的数据
df = (
    GowallaDataset().load(return_as="pl")
    .filter(pl.col("user_id").cast(pl.Int64) < 5)
    .with_columns(pl.col("check_in_time").dt.strftime("%Y%m%d").alias("ds"))
    .group_by("user_id", "ds")
    .agg(pl.col("loc_id").sort_by(pl.col("check_in_time")).alias("loc_seq"))
    .filter(pl.col("loc_seq").list.len() >= 5)
    .select(pl.col("user_id").alias("id"), "loc_seq")
)
print(df.shape)
df.head()
```

```{code-cell} ipython3
# 将user id存储到entity_id字段里面
loc_seqs = [LocSeq(seq=loc_seq, entity_id=user_id) for user_id, loc_seq in df.iter_rows()]

# 构建tokenizer
tokenizer = LocSeqTokenizer.build(loc_seqs)

# 构建一个BaseArrowDataset
train_ds = LocSeqDataset.init_from_loc_seqs(loc_seqs)

# 可以看到有56条序列
train_ds
```

```{code-cell} ipython3
def collate_function(batch: LocSeqDataset, tokenizer: LocSeqTokenizer):
    """
    将LocSeqDataset转换为TULER需要的样本
    - 序列（含padding）
    - 序列长度
    - 标签（用户id）
    """
    seqs: List[torch.LongTensor] = []
    lengths: List[int] = []
    labels: List[int] = []

    # 取出我们需要的两列
    seq_col = batch.seq
    entity_id_col = batch.entity_id

    # 按行遍历batch
    for line_idx in range(len(batch)):
        # 将位置序列使用tokenizer编码，以torch.LongTenso的类型返回
        seqs.append(tokenizer.tokenize_loc_seq(seq_col[line_idx], return_as="pt"))
        # 记录序列的长度
        lengths.append(seqs[-1].shape[0])
        # 添加标签
        labels.append(int(entity_id_col[line_idx].as_py()))

    # 返回三元组
    return (
        # 对序列添加padding，padding的值就用tokenizer维护的.pad属性
        pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad),
        lengths,
        torch.LongTensor(labels),
    )
```

```{code-cell} ipython3
# 定义DataLoader
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=lambda x: collate_function(x, tokenizer))

# 读取一个batch看看
iterator = iter(train_loader)
next(iterator)
```

当然，我们还可以封装一下上面的三元组：

```{code-cell} ipython3
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class TULERSampleExample:
    seqs: torch.LongTensor
    lengths: List[int]
    labels: Optional[torch.LongTensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


def collate_function_v2(batch: LocSeqDataset, tokenizer: LocSeqTokenizer):
    """
    将collate_function返回值的三元组封装成一个TULERSampleExample
    """
    seqs, lengths, labels = collate_function(batch, tokenizer)
    
    # 返回TULERSampleExample
    return TULERSampleExample(
        # 对序列添加padding，padding的值就用tokenizer维护的.pad属性
        seqs=seqs,
        lengths=lengths,
        labels=labels,
    )
```

```{code-cell} ipython3
# 定义DataLoader
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=lambda x: collate_function_v2(x, tokenizer))

# 读取一个batch看看
iterator = iter(train_loader)
batch = next(iterator)
batch
```

```{code-cell} ipython3
batch.batch_size
```

```{code-cell} ipython3
type(batch.seqs)
```

```{code-cell} ipython3
type(batch.lengths)
```

```{code-cell} ipython3
type(batch.labels)
```

```{tip}

`TrajDL`内支持的算法目前都已经提供了像上面`TULERSampleExample`一样的样本类，放在`trajdl.common.samples`目录下。

有一些算法需要的输入很多，比如5、6个参数，使用上述这样的样本类管理会比较清晰。

```

```{code-cell} ipython3

```
