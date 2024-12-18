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

# Quick Start

+++

`TrajDL`是一个基于Python的用于处理时空序列数据的深度学习算法包。`TrajDL`的目标是为科研人员、企业的工程师还有开源社区提供一个易用且高性能的算法工具。任何人可以通过这个算法包快速开展时空序列数据上的实验和开发工作。

我们会以一个经典的`TULER`算法为例，展示如何使用`TrajDL`快速开展实验。

+++

首先我们会加载`Gowalla`数据集，`TrajDL`会检查是否存在该数据集的缓存，如果不存在会下载原始数据集。

```{code-cell} ipython3
from tqdm.notebook import tqdm

from trajdl.datasets.open_source import GowallaDataset

# 下载并加载Gowalla数据集
df = GowallaDataset().load(return_as="pl")

df.head()
```

这里我们用一段简单的代码将数据集分成训练集和验证集

```{code-cell} ipython3
import polars as pl
import numpy as np

# 取id小于200的用户的数据
df = (
    df
    .filter(pl.col("user_id").cast(pl.Int64) < 200)
    .with_columns(pl.col("check_in_time").dt.strftime("%Y%m%d").alias("ds"))
    .group_by("user_id", "ds")
    .agg(pl.col("loc_id").sort_by(pl.col("check_in_time")).alias("loc_seq"))
    .filter(pl.col("loc_seq").list.len() >= 5)
    .select(pl.col("user_id").alias("id"), "loc_seq")
)

# 构建一个user_map，key是user的id（字符串类型），value是user id转换后的下标（int类型）
user_map = {
    user_id: idx for idx, user_id in enumerate(df.select(pl.col("id").unique())["id"])
}

# 随机取出80%的样本训练，20%验证
df = df.with_columns(pl.lit(np.random.uniform(size=(df.height,))).alias("random"))
train_df = df.filter(pl.col("random") <= 0.8).select("id", "loc_seq")
val_df = df.filter(pl.col("random") > 0.8).select("id", "loc_seq")

# 打印训练集和验证集的样本数，一行是一个样本
train_df.shape, val_df.shape
```

由于数据集是位置序列，我们需要基于数据构建一个`Tokenizer`将位置id转换为下标。

```{code-cell} ipython3
from trajdl.datasets import LocSeq
from trajdl.tokenizers import LocSeqTokenizer

# 将训练集和验证集的dataframe转换为List[LocSeq]，然后使用训练集的样本构建tokenizer
train_locseqs = [
    LocSeq(seq=loc_seq, entity_id=user_id) for user_id, loc_seq in train_df.iter_rows()
]
val_locseqs = [
    LocSeq(seq=loc_seq, entity_id=user_id) for user_id, loc_seq in val_df.iter_rows()
]

tokenizer = LocSeqTokenizer.build(loc_seqs=train_locseqs)
```

然后我们构建一个数据模块，数据模块是用来装载训练集、验证集、测试集的模块，可以提供高效的数据加载能力。

```{code-cell} ipython3
from trajdl.datasets import LocSeqDataset
from trajdl.datasets.modules.tuler import TULERDataModule

# 这里我们需要将List[LocSeq]转换为训练专用的批数据集LocSeqDataset
train_ds = LocSeqDataset.init_from_loc_seqs(train_locseqs)
val_ds = LocSeqDataset.init_from_loc_seqs(val_locseqs)

# 构建数据模块，这个数据模块提供了快速加载模型训练样本的功能
data_module = TULERDataModule(
    tokenizer=tokenizer,
    train_table=train_ds,
    val_table=val_ds,
    train_batch_size=32,
    val_batch_size=256,
    user_map=user_map,
    num_cpus=-1,
)
```

接下来构建一个`TULER`模型。

```{code-cell} ipython3
from trajdl.algorithms.tuler import TULER

# 构建模型，我们使用默认参数，用户也可以根据文档修改模型的类型，比如使用GRU、LSTM等编码器
model = TULER(
    tokenizer=tokenizer, num_users=len(user_map), embedding_dim=32, hidden_dim=32
)
```

最后我们使用`Lightning`训练，这里为了演示，训练的最大轮数设定为了3。

```{code-cell} ipython3
import lightning as L

# 模型训练，设定最大训练轮数为3，到3就停止，这里为了演示我们关闭lightning自身的logger和checkpoint
trainer = L.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
trainer.fit(model, data_module)
```

因为我们这里只设定了最大轮数为3，而且没有做location的预训练，所以准确率会低一些，这里只是演示如何快速训练一个`TULER`。

+++

接下来我们会演示如何推理一个样本，我们取验证集里面的第一条序列作为需要推理的样本。

```{code-cell} ipython3
import torch
from trajdl.common.samples import TULERSample

# 取验证集的第一条序列来演示如何推理
test_seq = val_locseqs[0]

# 通过tokenizer将序列进行编码，以torch.LongTensor的形式返回
src = tokenizer.tokenize_loc_seq(test_seq, return_as="pt").unsqueeze(dim=0)
print(f"src: {src}")

# 获得序列的长度
lengths = [src.shape[1]]
print(f"lengths: {lengths}")

# 这里使用TrajDL提供的Sample类型将样本组装起来
sample = TULERSample(src=src, seq_len=lengths)

with torch.inference_mode():
    # 这里实际上使用的是模型的forward方法，可以参照文档对照forward的参数及返回值
    output = model.eval()(sample)

    # 我们取出分数最大的那个样本的下标
    prediction_idx = torch.argmax(output, dim=-1).item()

    # 将user_map反转，key是下标，value是user_id
    user_map_reversed = {idx: user_id for user_id, idx in user_map.items()}

    # 获得预测出的user_id
    prediction = user_map_reversed[prediction_idx]

    # 序列的实际user_id
    ground_truth = test_seq.entity_id

    print(f"pred user id: {prediction}, ground_truth: {ground_truth}")
```

以上就是基于`TrajDL`的`TULER`快速实现。为了获得更好的效果，可以尝试扩大样本量、扩大训练轮数、使用预训练的位置嵌入等方法。

```{code-cell} ipython3

```
