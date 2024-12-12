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

# Lightning DataModule

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

```

`LightningDataModule`是`Lightning`里面的一个重要组成部分。使用`Lightning`框架训练的时候需要定义一个`LightningDataModule`用来管理数据。

```{seealso}
用户可以自行查阅[`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)的官方文档来了解`LightningDataModule`的作用。
```

`TrajDL`提供了一个适配**序列数据**和**轨迹数据**的抽象子类`BaseSeqDataModule`。这个`LightningDataModule`已经针对`TrajDL`的`BaseArrowDataset`提供了一些基础功能的封装，针对**序列数据**提供了`BaseLocSeqDataModule`，针对**轨迹数据**提供了`BaseTrajectoryDataModule`。用户可以快速基于这两个子类进行`LightningDataModule`的构建。

简单来说具体使用流程是：
1. 根据任务判断需要使用的是**位置序列**还是**轨迹序列**。
1. 根据数据集快速构建训练、验证、测试集，这些数据集使用`BaseArrowDataset`进行存储。
2. 根据任务编写`collate_function`。
3. 选择`BaseSeqDataModule`在其基础上编写自己的`LightningDataModule`，一般只要实现抽象方法的`collate_function`即可。

+++

我们以`TULER`为例快速实现一个`LightningDataModule`，我们要完成如下步骤：
1. 判断数据类型：使用`Gowalla`数据集，用的是**位置序列**。
2. 基于`Polars`快速构建训练、验证、测试集，构建Tokenizer
4. 编写`collate_function`。
5. 基于`BaseLocSeqDataModule`编写`TULERDataModuleExample`。

+++

## 构建训练、验证、测试集、Tokenizer

```{code-cell} ipython3
from tqdm.notebook import tqdm
import polars as pl
import numpy as np

from trajdl.datasets.open_source import GowallaDataset

# 下载并加载Gowalla数据集
df = GowallaDataset().load(return_as="pl")

# 取id小于20的用户的数据，为了演示，我们把序列的长度也设置一下
df = (
    df
    .filter(pl.col("user_id").cast(pl.Int64) < 20)
    .with_columns(pl.col("check_in_time").dt.strftime("%Y%m%d").alias("ds"))
    .group_by("user_id", "ds")
    .agg(pl.col("loc_id").sort_by(pl.col("check_in_time")).alias("loc_seq"))
    .filter((pl.col("loc_seq").list.len() >= 5) & (pl.col("loc_seq").list.len() < 10))
    .select("user_id", "ds", "loc_seq")
)
df.head()
```

```{code-cell} ipython3
# 构建一个user_map，key是user的id（字符串类型），value是user id转换后的下标（int类型）
user_map = {
    user_id: idx for idx, user_id in enumerate(df.select(pl.col("user_id").unique())["user_id"])
}
user_map
```

```{code-cell} ipython3
# 添加一列叫sample_idx，表示当前序列的日期在这个用户所有序列里面的排名
add_sample_idx = df.with_columns(sample_idx=pl.int_range(pl.len()).over("user_id", order_by="ds"))

# 统计每个用户的位置序列数
num_locseqs_by_user = df.group_by("user_id").agg(pl.len().alias("num_locseqs"))

# 使用join，将每个用户的序列数join到第一个DataFrame上
tmp_df = add_sample_idx.join(num_locseqs_by_user, how="left", on=["user_id"])

tmp_df.head()
```

```{code-cell} ipython3
# 针对每个用户，按ds划分训练、验证、测试集，比例是6: 2: 2
train_df = tmp_df.filter(pl.col("sample_idx") < pl.col("num_locseqs") * 0.6).select("user_id", "loc_seq")
val_df = tmp_df.filter((pl.col("sample_idx") >= pl.col("num_locseqs") * 0.6) & (pl.col("sample_idx") < pl.col("num_locseqs") * 0.8)).select("user_id", "loc_seq")
test_df = tmp_df.filter(pl.col("sample_idx") >= pl.col("num_locseqs") * 0.8).select("user_id", "loc_seq")

# 打印训练集，验证集和测试集的样本数，一行是一个样本
train_df.shape, val_df.shape, test_df.shape
```

```{code-cell} ipython3
# 使用LocSeqDataset装载三个数据集，构建Tokenizer
from trajdl.datasets import LocSeq, LocSeqDataset
from trajdl.tokenizers import LocSeqTokenizer


def transform_dataframe_into_dataset(df: pl.DataFrame) -> LocSeqDataset:
    """
    将一个Polars DataFrame转换为LocSeqDataset
    """
    locseqs = [LocSeq(seq=loc_seq, entity_id=user_id) for user_id, loc_seq in df.iter_rows()]
    return LocSeqDataset.init_from_loc_seqs(locseqs)


# 构建三个数据集
train_ds = transform_dataframe_into_dataset(train_df)
val_ds = transform_dataframe_into_dataset(val_df)
test_ds = transform_dataframe_into_dataset(test_df)

print("datasets:", train_ds, val_ds, test_ds)

# iter_as_seqs方法可以将BaseArrowDataset转换为单条序列的实例
tokenizer = LocSeqTokenizer.build(train_ds.iter_as_seqs())
```

## 编写collate_function

```{code-cell} ipython3
import torch
from typing import List, Optional, Dict
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from trajdl.common.samples import TULERSample


def collate_function(batch: LocSeqDataset, user_map: Dict[str, int], tokenizer: LocSeqTokenizer) -> TULERSample:
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
        # 添加标签，这里要用user_map将用户的id转换为idx
        labels.append(user_map[entity_id_col[line_idx].as_py()])

    # 返回样本
    return TULERSample(
        # 对序列添加padding，padding的值就用tokenizer维护的.pad属性
        src=pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad),
        seq_len=lengths,
        labels=torch.LongTensor(labels),
    )
```

## 编写LightningDataModule

因为我们使用的是**位置序列**，所以选择`BaseLocSeqDataModule`作为我们基座。

`BaseLocSeqDataModule`已经定义好了一些参数。

```{code-cell} ipython3
from trajdl.datasets.modules.abstract import BaseLocSeqDataModule

help(BaseLocSeqDataModule.__init__)
```

- tokenizer：`Tokenizer`，这个直接传入即可
- train_parquet_path, val_parquet_path, test_parquet_path：可选参数，这些是数据集的路径，`BaseArrowDataset`有一个`save`方法可以将数据集存储到文件，会以`parquet`的格式存储。
- train_table, val_table, test_table：可选参数：这些是数据集实例，也就是`BaseSeqDataModule`是同时支持传入文件路径或直接传入`BaseArrowDataset`实例进行数据集配置的。
- train_batch_size, val_batch_size：训练集和验证集的batch_size，测试集的batch_size会使用`val_batch_size`。
- train_sampler, val_sampler：可选参数：是否要传入`Sampler`，测试集不会使用`Sampler`。
- num_cpus：可选参数，有多少个CPU就会将`DataLoader`设置为多少个进程。

我们基于这个基类构建我们的`TULERDataModuleExample`，只要补充一个额外的`user_map`参数就好了，因为在collate_function里面要构建标签。

```{code-cell} ipython3
@dataclass
class TULERDataModuleExample(BaseLocSeqDataModule):
    user_map: Optional[Dict[str, int]] = None

    def __post_init__(self):
        # 先调用父类的后处理，因为使用的是dataclass，所以需要做这一步
        super().__post_init__()

        # 检查一下user_map这个参数的类型
        if not isinstance(self.user_map, dict):
            raise ValueError(
                "`user_map` should be a Dict[str, int] instance."
            )

    # 这个collate_function是一个抽象方法，子类必需实现
    def collate_function(self, batch: LocSeqDataset) -> TULERSample:
        # 因为父类已经存储了tokenizer，这里只要通过self.tokenizer即可获取
        return collate_function(batch, self.user_map, self.tokenizer)
```

这个`TULERDataModuleExample`就编写完了，很简单，实际上就是增加一个`user_map`的参数，然后再编写一个`collate_function`，接下来我们测试一下，训练集的batch_size设置为2，验证集设置为3。

```{code-cell} ipython3
datamodule = TULERDataModuleExample(
    tokenizer=tokenizer,
    train_table=train_ds,
    val_table=val_ds,
    test_table=test_ds,
    train_batch_size=2,
    val_batch_size=3,
    user_map=user_map)

datamodule.setup("fit")
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

```{code-cell} ipython3
next(iter(train_loader))
```

```{code-cell} ipython3
next(iter(val_loader))
```

```{code-cell} ipython3
next(iter(test_loader))
```

`BaseSeqDataModule`抽象了`TrajDL`在训练、验证、测试过程中数据的pipeline:
- 其提供了训练集、验证集、测试集的统一加载方式，也就是用户通过对数据集加工得到`BaseArrowDataset`后，可以直接放入`BaseSeqDataModule`里面，或者持久化之后又`BaseSeqDataModule`自动加载
- 提供训练、验证、测试集的batch_size的配置，提供`Sampler`的支持，提供`Tokenizer`的管理
- 自动加载`BaseArrowDataset`并构建`DataLoader`
- 用户在继承其子类（`BaseLocSeqDataModule`，`BaseTrajectoryDataModule`）的时候只要增加一些参数和自定义的`collate_function`即可。

```{tip}

`LightningDataModule`并不是必需使用的，因为`LightningDataModule`只是一个`DataLoader`的管理工具。用户可以根据自己的喜好自行定义训练流程，比如使用`Pytorch`原生的训练流程、或者使用`Lightning Fabric`，这些方式都可以自己定义`DataLoader`，不受`TrajDL`的约束。

```

```{code-cell} ipython3

```
