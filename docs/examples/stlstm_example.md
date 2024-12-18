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

# STLSTM

本节将会介绍使用`TrajDL`来复现`ST-LSTM`算法及在`Gowalla`数据集下的实验，主要包含如下内容：

* `Gowalla`数据集的预处理
* 基于`TrajDL`的`ST-LSTM`模型训练和推理

```{note}
`ST-LSTM`作者并未提供源码，`TrajDL`基于论文描述在Pytorch框架下进行了模型和实验的复现。在`TrajDL`中实现了`STLSTM`的完整实验流程，`HST-LSTM`会在后续版本中完成。
```


## Gowalla数据集预处理

在[Open Source Dataset](../data/open_source_dataset.md)中我们已经介绍了`Gowalla`数据集是一种基于社交网络的签到(Check-in)数据集，原始数据集的每一条记录都是一个位置ID，同时还有该位置ID的经纬度坐标。在[Quick Start](../getting_started/quick_start.md)中我们已经介绍过如何使用`TrajDL`中的接口来加载`Gowalla`数据集并将其转换为`Trajectory`和`LocSeq`。这些数据预处理的操作在本章中不再赘述。

在ST-LSTM中，还要额外处理轨迹序列两个相邻位置之间的*时间间隔*（time intervals）和*距离间隔*（distance intervals）。下面将通过代码来介绍如何进行数据的预处理。

先加载`Gowalla`数据集，并且提取出其中的Check-in轨迹序列：

```{code-cell} ipython3
from trajdl.datasets.open_source import GowallaDataset
import polars as pl


original_df = (
    GowallaDataset()
    .load(return_as="pl")
    .sort(["user_id", "check_in_time"])
    .with_columns(ds=pl.col("check_in_time").dt.strftime("%Y%m%d"))
    .with_columns(tmp_id=pl.int_range(pl.len()).over("user_id", "ds"))
    .limit(50000)
)
original_df.head(7)
```

计算每两个相邻位置之间的移动时间差：

```{code-cell} ipython3
df_with_ts = original_df.with_columns(
    ts_delta=(
        pl.col("check_in_time") - pl.col("check_in_time").shift(1)
    ).dt.total_seconds()
).with_columns(
    ts_delta=pl.when(pl.col("tmp_id") == 0)
    .then(pl.lit(0))
    .otherwise(pl.col("ts_delta"))
)
df_with_ts.head(7)
```

计算每两个相邻位置之间的移动距离差：

```{code-cell} ipython3
import math


df_with_dis = (
    df_with_ts.with_columns(
        lat1=pl.col("lat") * math.pi / 180, lng1=pl.col("lng") * math.pi / 180
    )
    .with_columns(lat2=pl.col("lat1").shift(1), lng2=pl.col("lng1").shift(1))
    .with_columns(
        dlat=(pl.col("lat2") - pl.col("lat1")),
        dlng=(pl.col("lng2") - pl.col("lng1")),
    )
    .with_columns(
        a=((pl.col("dlat") / 2).sin() ** 2)
        + pl.col("lat1").cos()
        * pl.col("lat2").cos()
        * ((pl.col("dlng") / 2).sin() ** 2)
    )
    .with_columns(dis_delta=2 * (pl.col("a") ** 0.5).arcsin() * 6371)
    .with_columns(
        dis_delta=pl.when(pl.col("tmp_id") == 0)
        .then(pl.lit(0))
        .otherwise(pl.col("dis_delta"))
    )
)

# 删掉无用的中间列
df_processed = df_with_dis.drop("lat1", "lat2", "lng1", "lng2", "dlat", "dlng", "a", "tmp_id")
df_processed.head(7)
```

以“天”为单位来切分访问会话（Visit Session），仅保留序列长度大于3且小于等于10的序列：

```{code-cell} ipython3
sessions = (
    df_processed.group_by("user_id", "ds")
    .agg(
        pl.len().alias("seq_len"),
        pl.col("loc_id").sort_by("check_in_time").alias("loc_seq"),
        pl.col("ts_delta").sort_by("check_in_time").alias("ts_delta"),
        pl.col("dis_delta").sort_by("check_in_time").alias("dis_delta"),
        (pl.col("check_in_time").min().dt.timestamp() / 1e6).alias("start_ts"),
    )
    .filter(pl.col("seq_len") > 3)
    .filter(pl.col("seq_len") <= 10)
    .select("user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts")
)
sessions.head(5)
```

因为在ST-LSTM中需要以Visit Session为单位来计算每个Visit Session的特征，所以此处先计算每个用户下的Visit Session的数量：

```{code-cell} ipython3
tmp_df = (
    sessions.sort(["user_id", "start_ts"])
    .with_columns(
        tmp_id=pl.int_range(pl.len()).over("user_id", order_by="start_ts")
    )
    .join(
        sessions.group_by("user_id").agg(pl.len().alias("num_sessions")),
        how="left",
        on=["user_id"],
    )
)
tmp_df.head(5)
```

切分训练集:验证集:测试集=6:2:2，以Visit Session为粒度进行划分

```{code-cell} ipython3
# 划分训练集:验证集:测试集=6:2:2
train_df = tmp_df.filter(pl.col("tmp_id") <= pl.col("num_sessions") * 0.6).select(
    "user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts"
)

val_df = tmp_df.filter(
    (pl.col("tmp_id") > pl.col("num_sessions") * 0.6)
    & (pl.col("tmp_id") <= pl.col("num_sessions") * 0.8)
).select("user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts")

test_df = tmp_df.filter(pl.col("tmp_id") > pl.col("num_sessions") * 0.8).select(
    "user_id", "loc_seq", "ts_delta", "dis_delta", "start_ts"
)
```

分别构建训练、验证和测试的`LocSeqDataset`:

```{code-cell} ipython3
from tqdm.notebook import tqdm
from trajdl.datasets import LocSeq, LocSeqDataset

def generate_loc_seqs(df):
    return [
        LocSeq(
            seq=loc_seq,
            entity_id=user_id,
            ts_delta=ts_delta,
            dis_delta=dis_delta,
            start_ts=start_ts,
        )
        for user_id, loc_seq, ts_delta, dis_delta, start_ts in tqdm(
            df.iter_rows(), total=df.height
        )
    ]

train_ds, val_ds, test_ds = (
    LocSeqDataset.init_from_loc_seqs(generate_loc_seqs(train_df)),
    LocSeqDataset.init_from_loc_seqs(generate_loc_seqs(val_df)),
    LocSeqDataset.init_from_loc_seqs(generate_loc_seqs(test_df)),
)
```

构建LocSeqTokenizer并保存，此处在`Gowalla`数据集中每个Check-in的位置都有一个唯一的ID，所以无需使用`TrajDL`中的`GridSystem`来做`Trajectory`的离散化。

```{code-cell} ipython3
from trajdl.tokenizers import LocSeqTokenizer

train_locseqs = generate_loc_seqs(train_df)
tokenizer = LocSeqTokenizer.build(loc_seqs=train_locseqs)
```

计算时间间隔和距离间隔的上下界，并构建`Bucketizer`。给定一个数或者一组数，`Bucketizer`可以调用`get_bucket_index`或者`get_bucket_indices`函数来计算其对应的bucket，此处的bucket也就是论文中的`slot`切片。

```{code-cell} ipython3
# 计算数据集的ts_delta的上下界
tsd_stats = train_df.select(ts_delta=pl.col("ts_delta").explode()).select(
    ts_delta_max=pl.col("ts_delta").max(), ts_delta_min=pl.col("ts_delta").min()
)

# 计算数据集dist_delta的上下界
disd_stats = train_df.select(dis_delta=pl.col("dis_delta").explode()).select(
    dis_delta_max=pl.col("dis_delta").max(), dis_delta_min=pl.col("dis_delta").min()
)

ts_delta_lower, ts_delta_upper = math.floor(tsd_stats["ts_delta_min"][0]), math.ceil(
    tsd_stats["ts_delta_max"][0]
)
dis_delta_lower, dis_delta_upper = math.floor(
    disd_stats["dis_delta_min"][0]
), math.ceil(disd_stats["dis_delta_max"][0])

print(ts_delta_lower, ts_delta_upper, dis_delta_lower, dis_delta_upper)
```

构建`Bucketizer`：

```{code-cell} ipython3
from trajdl.tokenizers.slot import Bucketizer

time_bucketizer = Bucketizer(
    lower_bound=ts_delta_lower, upper_bound=ts_delta_upper, num_buckets=10
)
loc_bucketizer = Bucketizer(
    lower_bound=dis_delta_lower, upper_bound=dis_delta_upper, num_buckets=10
)
```

## DataModule

```{code-cell} ipython3
from trajdl.datasets.modules.stlstm import STLSTMDataModule

data_module = STLSTMDataModule(
    tokenizer=tokenizer,
    train_table=train_ds,
    val_table=val_ds,
    test_table=test_ds,
    ts_bucketizer=time_bucketizer,
    loc_bucketizer=loc_bucketizer,
    train_batch_size=4,
    val_batch_size=4,
    num_train_batches=50,
    num_val_batches=20,
    num_cpus=-1,
)

data_module.setup("fit")
train_dataloader = data_module.train_dataloader()
next(iter(train_dataloader))
```

## ST-LSTM模型

```{code-cell} ipython3
from trajdl.algorithms.loc_pred.stlstm import STLSTMModule

# 构建GM-VASE模型
model = STLSTMModule(
    tokenizer=tokenizer,
    embedding_dim=128,
    hidden_size=256,
    ts_bucketizer=time_bucketizer,
    loc_bucketizer=loc_bucketizer,
)
model
```

## 训练

执行如下的代码即可开始模型的训练：

```{code-cell} ipython3
import lightning as L

trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
trainer.fit(model, data_module)
```

## 推理

在`STLSTMModule`中的`forward`函数即为推理函数，其输出即为预测的下一位置的概率，经过`argmax`操作之后即为概率最大的下一位置预测：

```{code-cell} ipython3
import torch

data_module.setup("test")
test_loader = data_module.test_dataloader()


with torch.inference_mode():
    predictions = trainer.predict(model, test_loader)

print(predictions[0].argmax(dim=-1))
```


```{tip}
1. 本文介绍了ST-LSTM算法的数据集预处理、时空特征处理以及模型的训练&推理。
2. 本文代码中大量使用`Polars`来处理数据，推荐读者在使用`TrajDL`时同样使用`Polars`工具进行数据处理。
```
