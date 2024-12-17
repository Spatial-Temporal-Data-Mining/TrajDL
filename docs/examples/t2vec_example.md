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

# T2VEC

在本篇文档中，我们将会使用`TrajDL`来实现T2VEC算法。我们会以代码的形式介绍如下内容：

* Porto轨迹数据集的加载和预处理
* 原始数据集转化为TrajDL标准数据集
* Tokenizer构建
* T2VEC模型训练和推理


## Porto轨迹数据加载和预处理

[Porto数据集]是波尔多市出租车轨迹数据，详细介绍见[Open Source Datasets](../data/open_source_dataset.md)。在这里我们将使用`TrajDL`的`PortoDataset`接口来加载。

```{code-cell} ipython3
import polars as pl
from trajdl.datasets.open_source import PortoDataset

# 以polar.DataFrame的数据类型返回，为了便于展示，限制10000条数据
porto = PortoDataset()

original_trajs = (
    porto.load(return_as="pl")
    .filter(pl.col("MISSING_DATA") == False)
    .sort("TIMESTAMP")["POLYLINE"]
    .limit(10000)
)

original_trajs.head(1)
```

```{tip}
`TrajDL`还提供可扩展的`OpenSourceDataset`的接口来支持用户自行配置和加载开源数据集。
```


## TrajectoryDataset的构建

加载好Porto数据集之后，还无法直接用于模型的输入，需要将其转化为`TrajDL`中的标准轨迹点序列数据`Trajectory`，下面将介绍如何使用`TrajDL`中的API进行转换。

```{code-cell} ipython3
from tqdm.contrib import tenumerate
from trajdl.datasets import Trajectory

all_trajs = [
    Trajectory(traj_pl.to_numpy(), entity_id=str(idx))
    for idx, traj_pl in tenumerate(original_trajs, desc="transform trajectorys")
]
print(all_trajs[0], all_trajs[0].seq[:10])
```

接下来，我们将`Trajectory`切分为训练集、验证集和测试集部分。我们设定训练集800条轨迹，验证集和测试集分别是100条轨迹，序列最短长度是20，最大长度是100。

```{code-cell} ipython3
from tqdm.notebook import trange

# 定义超参数
NUM_TRAIN, NUM_VAL, TEST_START_IDX, NUM_TEST = 800, 100, 900, 100
MIN_LENGTH, MAX_LENGTH = 20, 100
MIN_LENGTH_TEST, MAX_LENGTH_TEST = 60, 200

all_trajs = all_trajs[:1000]

train_traj, val_traj = [], []
for idx in trange(NUM_TRAIN + NUM_VAL, desc="construct train and val set"):
    traj = all_trajs[idx]
    if MIN_LENGTH <= len(traj) <= MAX_LENGTH:
        if idx <= NUM_TRAIN:
            train_traj.append(traj)
        else:
            val_traj.append(traj)

test_traj = []
for idx in trange(TEST_START_IDX, len(all_trajs), desc="construct test set"):
    traj = all_trajs[idx]
    if len(test_traj) >= NUM_TEST:
        break
    if MIN_LENGTH_TEST <= len(traj) <= MAX_LENGTH_TEST:
        test_traj.append(traj)

print(len(train_traj), len(val_traj), len(test_traj))
```

```{tip}
如果数据集过于庞大，亦可以将其转换为`TrajectoryDatset`并使用`save`函数来将数据集保存在磁盘中。
```

### 网格系统

对于上文得到的`TrajectoryDataset`，其中数据的基本形式是轨迹的经纬度点序列，我们需要将其离散化转换为类似于自然语言处理领域的`token`，后续作为`Embedding Layer`的输入来获取词向量嵌入表征。

因此，在`TrajDL`中我们提供了`GridSystem`网格系统的API，读者使用该API可直接建立一套用于经纬度点转网格的系统，将连续的经纬度点离散化为网格（网格即为`token`）。另外，`GridSystem`底层封装了`C++`编写的`trajdl_cpp`工具来优化计算，详细的介绍参见[网格系统](../tools/grid.md)。

波尔多市的地图如[图1](#porto-map)所示，我们按照$100m \times 100m$为一个网格将地图进行切分，以此构建网格系统。

```{figure} ../_static/images/porto_google_map.jpg
:alt: Porto Map
:width: 500px
:align: center
:name: porto-map

图1. 波尔多市地图（图源：Google Map）
```

先创建网格系统在波尔多市的边界

```{code-cell} ipython3
from trajdl import trajdl_cpp

# 基于经纬度系统的区域边界
boundary_original = trajdl_cpp.RectangleBoundary(
    min_x=-8.690261,
    min_y=41.140092,
    max_x=-8.549155,
    max_y=41.185969,
)
# 转换为基于平面坐标系的区域边界
boundary = boundary_original.to_web_mercator()
print(f"boundary_original: {boundary_original}")
print(f"boundary: {boundary}")
```

通过打印的`boundary_original`和`boundary`两个变量可以看出这里存在两个坐标系统，一种是基于原始经纬度的坐标系统，另外一种是墨卡托投影系统，将原始的三维地球表面映射到二维平面系统。在T2VEC中使用的是墨卡托投影系统，该系统中数值的单位是”米“，在赤道处的比例是$1:1$（即墨卡托系统里赤道处的1米对应于1米的实际距离）。更为详细的介绍参见[Grid](../tools/grid.md)。

在`TrajDL`中提供了将经纬度坐标转换为平面坐标的API：

```{code-cell} ipython3
print(trajdl_cpp.convert_gps_to_webmercator(-8.690261, 41.140092))
print(trajdl_cpp.convert_gps_to_webmercator(0, 0))
```

可以看到经纬度坐标$(0,0)$对应墨卡托投影系统中的坐标原点$(0,0)$。下面我们在墨卡托系统中对波尔多市的地图进行网格系统的构建，一个网格单元的大小是$100m \times 100m$

```{code-cell} ipython3
from trajdl.grid import SimpleGridSystem

# 网格的划分距离为100m
grid_width, grid_height = 100, 100

# 创建网格系统
grid = SimpleGridSystem(
    # 使用波尔多市的左下角点和右上角点来构建和切分网格系统
    boundary,
    step_x=grid_width,
    step_y=grid_height,
)
print(len(grid), grid.num_x_grids, grid.num_y_grids)
```

给定一个波尔多市的经纬度，即可映射到`SimpleGridSystem`中的一个网格`token`

```{code-cell} ipython3
# 转墨卡托坐标系统
web_mercator_location = trajdl_cpp.convert_gps_to_webmercator(-8.610291, 41.140746)
# 转网格id
x, y = web_mercator_location.x, web_mercator_location.y
grid_id = grid.locate_unsafe(x, y)

print(web_mercator_location, grid_id)
```

## Tokenizer的构建

在构建完`GridSystem`之后，我们要开始构建`Tokenizer`。在此处，`Tokenizer`的作用是给定一个输入的经纬度点（对应自然语言处理领域中一个`word`），使用`Tokenizer`将其转换为一个`token`，该`token`的表现形式是一个`int`类型的整数，作为`Embedding Layer`的输入来得到对应的*词嵌入表示*（word embedding）。

在`TrajDL`中，我们提供了`T2VECTokenizer`的API来构建`Tokenizer`：

```{code-cell} ipython3
import os

from trajdl.tokenizers.t2vec import T2VECTokenizer

output_folder = "./output/t2vec"
os.makedirs(output_folder, exist_ok=True)

tokenizer = T2VECTokenizer.build(
    grid=grid,
    boundary=boundary_original,
    trajectories=all_trajs,
    max_vocab_size=40000,  # 词表支持的词元上限，排序逻辑是词元在数据集中的频率
    min_freq=100,  # 被命中至少min_freq次的网格称之为`hot cell`，tokenizer中仅保留`hot cell`
    with_kd_tree=True,
)
tokenizer.save_pretrained(os.path.join(output_folder, "tokenizer.pkl"))
print("num vocab: ", len(tokenizer))
```

在这里，解释一下`min_freq`和`with_kd_tree`这两个参数的意义：

* `Tokenizer`构建的词表越大，那么其后续的计算量越大复杂度越高，所以希望能够用最小的词表来表示整个数据集，所以在此处，就有`min_freq`这个参数来限制，当某个词元出现的频率小于`min_freq`的时候，`Tokenizer`会剔除掉该词元，以此方式来平衡词表的大小和词表的信息量。
* 基于上一点，如果轨迹序列的经纬度点命中了一些被剔除掉的网格`cell`，那么`Tokenizer`无法将该轨迹点转换为`token`。此时，我们提供了`KDTree`搜索的方法，将所有的`hot cell`构建`KDTree`，当轨迹点没有命中`hot cell`时，则会通过`KDTree`来搜索与该轨迹点最相近的`hot cell`作为其近似`token`。


```{note}
在`T2VECTokenizer`中传入的`boundary`变量是基于原始经纬度的`boundary_original`。
```

## K近邻网格
在[T2VEC](../available_algorithms/t2vec.md)中，作者为了计算基于位置距离加权的损失函数，还构建了两个矩阵，分别是当前网格的10个近邻网格索引的矩阵 $V$ ( $N \times 10$ )和当前网格的10个近邻网格距离的矩阵 $D$ ( $N \times 10$ )。在tokenizer中也集成了这两个矩阵的构建函数`k_nearest_hot_loc`。

```{code-cell} ipython3
from trajdl.common.enum import TokenEnum

k = 10  # 10个最近的网格
SPECIAL_TOKENS = TokenEnum.values()
vocab_list = tokenizer.vocab.keys()  # 获取全部字典的Token
loc_list, idx_list = zip(
    *((loc, tokenizer.loc2idx(loc)) for loc in vocab_list if loc not in SPECIAL_TOKENS)
)  # 剔除special tokens字
```

```{code-cell} ipython3
import numpy as np

dists, locations = tokenizer.k_nearest_hot_loc(
    loc_list, k=k
)  # 获取k个最近的网格以及对应的距离

# (num_locations, k)，索引矩阵
V = np.zeros(shape=(len(vocab_list), k), dtype=np.int64)

# (num_locations, k)，距离矩阵
D = np.zeros_like(V, dtype=np.float32)
D[idx_list, :] = dists

# 对于SPECIAL TOKENS，最近的token设定为自己
for token in SPECIAL_TOKENS:
    idx = tokenizer.loc2idx(token)
    V[idx] = idx

for line_idx, loc_list in zip(idx_list, locations):
    V[line_idx] = [tokenizer.loc2idx(loc) for loc in loc_list]

np.save(os.path.join(output_folder, "knn_indices.npy"), V)  # 保存k近邻网格的索引
np.save(os.path.join(output_folder, "knn_distances.npy"), D)  # 保留k近邻网格的距离
```

## 训练和推理

`TrajDL`中训练环节是基于`Lightning`框架编写的，典型的模块比如数据模块`T2VECDataModule`是继承自`LightningDataModule`、模型模块`T2VEC`是基于`LightningModule`模块编写的，用户可以通过简单几行代码，调用`TrajDL`中的`API`进行训练和推理。

```{tip}
另外，`Lightning`框架还提供了命令行和配置文件的方式进行模型训练与验证。读者如果想对`Lightning`的命令行与配置文件有进一步的了解，可以阅读[LightningCLI](../training/lightning_cli.md)。
```

此处，我们仅展示使用`API`的方式来进行`T2VEC`模型的训练和推理。

首先，我们使用`TrajDL`导入数据模块`T2VECDataModuleV2`，在该模块中的`collate_function`中实现了`downsampling`和`distortion`两种样本增强的方式。

```{code-cell} ipython3
from trajdl.datasets import TrajectoryDataset
from trajdl.datasets.modules.t2vec import T2VECDataModuleV2

# 将Trajectory转换为TrajectoryDataset
train_dataset = TrajectoryDataset.init_from_trajectories(train_traj)
val_dataset = TrajectoryDataset.init_from_trajectories(val_traj)
test_dataset = TrajectoryDataset.init_from_trajectories(test_traj)

data_module = T2VECDataModuleV2(
    tokenizer=tokenizer,
    train_table=train_dataset,
    val_table=val_dataset,
    test_table=test_dataset,
    train_batch_size=4,
    val_batch_size=4,
    num_train_batches=10,
    num_val_batches=10,
    num_cpus=-1,
    k=2,
)
```

```{note}
1. 在上文中，我们使用了[`T2VECDataModuleV2`](../../src/trajdl/datasets/modules/t2vec.py#L195)来构建数据模块，该模块在[`collate_function`](../training/collate_function.md)中实现了样本下采样和扭曲的变换方式，在训练迭代的过程中，会在线生成对比的样本。此外，在`TrajDL`中还存在一种样本构建的方式，即离线样本构建，需要在训练之前将样本构建完毕存储到磁盘中，在训练时再从磁盘中加载样本到内存中，[`T2VECDataModule`](../../src/trajdl/datasets/modules/t2vec.py#L68)中的初始化接受的`src_path`和`trg_path`即为离线构建的对比样本在磁盘中的路径。
2. 此处为了演示`TrajDL`中API的使用，采用了在线样本构建的`T2VECDataModuleV2`，但是在Benchmark中，为了复现论文实验效果，使用了离线样本构建的`T2VECDataModule`，具体参见T2VEC的Benchmark代码即可。
```

执行如下代码可以得到训练数据集中第一个batch的数据

```{code-cell} ipython3
data_module.setup("fit")
train_dataloader = data_module.train_dataloader()
next(iter(train_dataloader))
```

```{note}
此处的`T2VECDataModuleV2`继承自`BaseTrajectoryDataModule`，该模块可以直接接受`TrajectoryDataset`作为初始化数据集。
```

使用`TrajDL`中的API可以直接导入模型：

```{code-cell} ipython3
from trajdl.algorithms.t2vec import T2VEC

# 构建模型，我们使用默认参数，用户也可以根据文档修改模型的类型，比如使用GRU、LSTM等编码器
model = T2VEC(
    embedding_dim=256,
    hidden_size=256,
    tokenizer=tokenizer,
    knn_indices_path=os.path.join(output_folder, "knn_indices.npy"),
    knn_distances_path=os.path.join(output_folder, "knn_distances.npy"),
)
model
```

使用`PytorchLightning`中的`Trainer`，两行代码即可开始模型的训练，训练的结果在文件夹`lightning_logs`中。为了展示训练的流程，在此处的代码中仅训练一轮，在一轮训练结束执行，会在验证数据集`val_dataset`上进行验证。

```{code-cell} ipython3
import lightning as L

trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
trainer.fit(model, data_module)
```

模型训练完成之后，即可开始在`test_dataset`上的推理，推理的代码如下：

```{code-cell} ipython3
import torch

# 先获取test_dataloader
data_module.setup("test")
test_dataloader = data_module.test_dataloader()
test_sample_1 = next(iter(test_dataloader))
test_sample_2 = next(iter(test_dataloader))

model.eval()

with torch.inference_mode():
    vec_1 = model(test_sample_1)
    vec_2 = model(test_sample_2)

batch_vec_1, batch_vec_2 = vec_1.detach().cpu().numpy(), vec_2.detach().cpu().numpy()
print(batch_vec_1.shape, batch_vec_2.shape)
```

计算两个batch中样本两两之间的轨迹相似度：

```{code-cell} ipython3
from sklearn.metrics.pairwise import euclidean_distances

print(euclidean_distances(batch_vec_1, batch_vec_2))
```

```{note}
1. 此处的API调用仅展示如何使用`TrajDL`来快速开展`T2VEC`模型的训练和推理，论文中的消融实验的设计并非这样直接对测试数据进行推理后计算Embedding的相似度。
2. 关于完整的实验，以及使用命令行和`YAML`配置文件的方式开展模型的训练和部署，可以参见 Benchmark中的内容，其中已经给出用于T2VEC复现的完整的配置文件和执行脚本，包括数据预处理、词嵌入模型预训练、模型的训练&部署、论文实验的复现（在代码中也给出了诸多灵活使用`PytorchLighting`的示例代码供以学习参考）。
```


```{tip}
1. 介绍了如果使用`TrajDL`中的`API`来进行T2VEC中数据集、tokenizer和增强样本的构建。
2. 基于`TrajDL`快速开展T2VEC的训练和推理。
```
