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

# GMVSAE

在本节内容中，我们主要是介绍使用`TrajDL`来进行轨迹序列异常检测，并且针对GM-VSAE中的部分公式进行代码细节讲解。本节内容如下：

* 数据预处理
* GM-VSAE训练
* 在线异常序列检测


## 数据预处理

在GM-VSAE中使用的数据集依旧是前文所介绍过的[Porto数据集](../data/open_source_dataset.md)，该数据集是从出租车上采集到的波尔多市出租车轨迹数据集。下面我们基于`TrajDL`中的网格系统将原始的经纬度点序列转换为网格位置序列。

这里我们使用了`SimpleGridSystem`构建网格系统，使用`RectangleBoundary`构建区域。

```{code-cell} ipython3
from trajdl import trajdl_cpp
from trajdl.grid import SimpleGridSystem

# 网格大小(0.1km, 0.1km)
grid_height, grid_width = 0.1, 0.1
lat_size, lng_size = grid_height / 110.574, grid_width / 111.320 / 0.99974
grid = SimpleGridSystem(
    # 使用波尔多市的左下角点和右上角点来构建和切分网格系统
    trajdl_cpp.RectangleBoundary(
        min_x=-8.690261,
        min_y=41.140092,
        max_x=-8.549155,
        max_y=41.185969,
    ),
    step_x=lng_size,
    step_y=lat_size,
)
print("Grid Size:", len(grid))
```

```{note}
注意，此处使用的坐标系统为基于原始经纬度的坐标系统，并非[T2VEC](./t2vec_example.md)中使用的墨卡托坐标系统。
```

下面使用`TrajDL`中的公开数据集接口`PortoDataset`来加载Porto数据集，并将轨迹点序列`Trajecotory`转换为位置序列`LocSeq`。

```{code-cell} ipython3
from collections import defaultdict

import polars as pl
from tqdm.notebook import tqdm
from tqdm.contrib import tenumerate
from trajdl.datasets import LocSeq
from trajdl.datasets.open_source.conf import PortoDataset


# 定义最短和最长的序列长度
shortest, longest = 20, 1200

def generate_od_map(shortest: int, longest: int):
    # 加载Porto数据集并以pl.DataFrame的数据类型返回
    trajectories = (
        PortoDataset()
        .load(return_as="pl")
        .select("POLYLINE")
        .filter(
            (pl.col("POLYLINE").list.len() >= shortest)
            & (pl.col("POLYLINE").list.len() <= longest)
        )["POLYLINE"]
        .limit(100000)
    )
    print(len(trajectories), trajectories[0])

    # 定义一个np.ndarray类型的轨迹点序列转换为LocSeq位置序列的函数
    def transform_traj_into_loc_seq(traj_np, idx):
        # 检测该序列是否所有点都在波尔多市的网格系统中
        if grid.in_boundary_np(traj_np).all():
            return LocSeq(grid.locate_unsafe_np(traj_np), entity_id=str(idx))
        return None

    # 基于OD构建一个dict，key是OD pair，values是位置序列组成的list
    od_agg = defaultdict(list)
    for idx, traj_pl in tenumerate(trajectories, desc="transform trajectories into location sequences"):
        # 将pl.DataFrame类型的原始序列转化为位置序列
        loc_seq = transform_traj_into_loc_seq(traj_pl.to_numpy(), idx)
        if loc_seq:
            od_agg[(loc_seq.o, loc_seq.d)].append(loc_seq)

    return od_agg


od_agg = generate_od_map(shortest, longest)
```

下面开始统计属于同一对起点和终点的轨迹序列数量，也即论文中的”固定行程“，并且划分训练集和验证集。划分的逻辑是，先统计有多少"起点-终点(OD)"的路程，然后统计每个路程中有多少条轨迹，过滤掉小于25条轨迹的路程。然后从每一条路程中，选择最后的5条序列作为验证集，其余的序列作为训练集。

```{code-cell} ipython3
# 划分训练和验证集
min_od_traj_num = 10
test_traj_num = 5

train_loc_seqs, val_loc_seqs, valid_ods = [], [], set()
for od, loc_seqs in tqdm(od_agg.items(), desc="generating dataset"):
    num_loc_seqs = len(loc_seqs)
    # 过滤掉小于25的OD路程
    if num_loc_seqs >= min_od_traj_num:
        for idx in range(num_loc_seqs - test_traj_num):
            train_loc_seqs.append(loc_seqs[idx])
        for idx in range(num_loc_seqs - test_traj_num, num_loc_seqs):
            val_loc_seqs.append(loc_seqs[idx])
        valid_ods.add(od)
print(len(train_loc_seqs), len(val_loc_seqs), len(valid_ods))
```

```{code-cell} ipython3
import numpy as np
from trajdl.datasets import LocSeqDataset


def generate_dataset(loc_seqs, nums):
    sample_idx = np.random.choice(
        list(range(len(loc_seqs))), size=nums, replace=False
    )
    loc_samples = [loc_seqs[i] for i in sample_idx]
    return LocSeqDataset.init_from_loc_seqs(loc_samples)


train_dataset, val_dataset = (
    generate_dataset(train_loc_seqs, 800),
    generate_dataset(val_loc_seqs, 200),
)
print(train_dataset, val_dataset)
```

```{note}
本节后续会展示GM-VASE模型在Porto数据集上训练和预训练，考虑到执行时间，此处在构建数据集的时候对训练序列`train_loc_seqs`采样800条数据，对验证序列`val_loc_seqs`采样200条序列。该参数用户在使用`TrajDL`中可自行设定。
```

在GM-VSAE中，因为没有异常轨迹的标签，所以在这里需要构建异常轨迹的序列，用于序列异常检测的下游任务。构建异常轨迹序列的方式是：从数据集中随机选取一定比例的轨迹序列，然后对轨迹序列上的每一个轨迹点进行随机扰动。扰动的方式是，随机从该点所在的位置选择其一阶邻居或二阶邻居的点作为该点的扰动进行替换。具体的实现如下：

```{code-cell} ipython3
# 先定义在GridSystem中扰动的函数
def perturb_locseq(locseq, level, prob):
    loc_list = [locseq.o]
    for idx in range(1, len(locseq) - 1):
        loc = locseq[idx]
        # 获取loc的邻域
        grid_x, grid_y = grid.to_grid_coordinate(loc)
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        if grid.in_boundary_by_grid_coordinate(
            grid_x + x_offset * level, grid_y + y_offset * level
        ):
            grid_x += x_offset * level
            grid_y += y_offset * level
        loc_list.append(
            grid.locate_by_grid_coordinate(grid_x, grid_y)
            if np.random.random() < prob
            else loc
        )

    loc_list.append(locseq.d)
    return LocSeq(seq=loc_list)
```

在下文中，该异常数据要用于在线推理，为了便于展示，仅选取100条数据作为测试数据。

```{code-cell} ipython3
# 对数据集进行扰动
rng = np.random.default_rng(seed=42)
# 扰动的范围在网格的二阶邻域内
level = 2
ratio = 0.1
point_prob = 0.3

tmp_loc_seqs = train_loc_seqs[:100]

num_train_trajs = len(tmp_loc_seqs)
print(f"num_train_trajs: {num_train_trajs}")

train_outlier_idx = rng.choice(
    num_train_trajs, int(num_train_trajs * ratio), replace=False
)
print(f"num outliers in training set: {train_outlier_idx.shape[0]}")

for outlier_idx in tqdm(train_outlier_idx):
    tmp_loc_seqs[outlier_idx] = perturb_locseq(
        tmp_loc_seqs[outlier_idx], level=level, prob=point_prob
    )

train_outliers_ds = LocSeqDataset.init_from_loc_seqs(tmp_loc_seqs)
```

## 构建tokenizer

接下来，使用`TrajDL`中提供的`LocSeqTokenizer`来构建该数据集的`tokenizer`，后续模型的训练会使用该`tokenizer`。

```{code-cell} ipython3
from trajdl.tokenizers.locseq import LocSeqTokenizer

tokenizer = LocSeqTokenizer.build(
    loc_seqs=train_loc_seqs, count_start_end_token=False
)
```

## 构建DataModule

`TrajDL`中已经封装好`GMVSAEDataModule`，直接导入并且实例化即可：

```{code-cell} ipython3
from trajdl.datasets.modules.gmvsae import GMVSAEDataModule

data_module = GMVSAEDataModule(
    tokenizer = tokenizer,
    train_table = train_dataset,
    val_table = val_dataset,
    test_table=train_outliers_ds,
    train_batch_size = 1,
    val_batch_size = 1,
    num_cpus = -1,
    num_train_batches = 5,
    num_val_batches = 5,
)

data_module.setup("fit")
train_dataloader = data_module.train_dataloader()
next(iter(train_dataloader))
```

## GM-VASE模型

使用`TrajDL`中的`GMVSAE`快速构建模型：

```{code-cell} ipython3
from trajdl.algorithms.gmvsae import GMVSAE
from trajdl.common.enum import Mode

# 构建GM-VASE模型
model = GMVSAE(
    tokenizer=tokenizer,
    embedding_dim=32,
    hidden_size=256,
    mem_num=10,
    mode=Mode.TRAIN,
)
model
```

## 训练

执行如下代码即可快速开展模型的训练：

```{code-cell} ipython3
import lightning as L

trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
trainer.fit(model, data_module)
```

## 推理

在推理部分，模型输出的是序列的异常分数。在上面我们已经介绍了我们使用扰动的方式构造了含有异常的轨迹序列数据`train_outliers_ds`以及对应的异常轨迹的索引`train_outliers_idx`，下面来对这部分数据进行推理。

```{code-cell} ipython3
import torch

# 获取train_outliers_ds的数据
data_module.setup("test")
test_loader = data_module.test_dataloader()

# 将模型设置为EVAL模式，在该模式下，forward函数输出轨迹序列的异常分数
model.set_mode(Mode.EVAL)

with torch.inference_mode():
    predictions = trainer.predict(model, test_loader)
predictions = np.concatenate([i.detach().cpu().numpy() for i in predictions])

predictions
```

使用下面的代码来计算AUC的值：

```{code-cell} ipython3
from sklearn.metrics import auc, precision_recall_curve


# 先定义一个计算AUC的函数的函数
def auc_score(y_true, y_score):
    # shape of precision and recall is (N - 1,)
    # 这里用1减去，说明0是正常序列，1是异常序列，precision_recall_curve默认的pos_label是1
    precision, recall, _ = precision_recall_curve(1 - y_true, 1 - y_score)
    # float
    return auc(recall, precision)


y_true = np.ones(shape=(len(train_outliers_ds)))
for idx in train_outlier_idx:
    y_true[idx] = 0

od_agg = defaultdict(list)
for idx, loc_seq in tenumerate(
    train_outliers_ds.iter_as_seqs(), total=len(train_outliers_ds)
):
    od_agg[(loc_seq.o, loc_seq.d)].append(idx)

od_auc = {}
for od, traj_indices in tqdm(od_agg.items()):
    od_true, od_pred = y_true[traj_indices], predictions[traj_indices]
    if od_true.sum() < od_true.shape[0]:
        od_auc[od] = auc_score(od_true, od_pred)

np.mean(list(od_auc.values()))
```

```{tip}
1. 本文介绍了GM-VSAE中数据预处理的方式。
2. 本文介绍了使用`TrajDL`搭建GM-VSAE模块并开展实验。
```
