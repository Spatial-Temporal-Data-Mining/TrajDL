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

# Open Source Datasets

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

```

TrajDL提供了一个公开数据集接口，可以帮助用户管理公开数据集，如Gowalla, porto等。用户可以通过这类接口快速下载原始数据集，`TrajDL`会构建缓存，提升后续数据集的加载效率。

下面我们会介绍这些公开数据集及其使用方法。

+++

## Gowalla数据集

[Gowalla](https://snap.stanford.edu/data/loc-gowalla.html) 是一个基于位置的社交网络网站，用户通过签到打卡的方式分享他们的位置。用户的每次签到都会记录其签到的时刻（Timestamp）、地点（Location）和GPS位置（Point），其中每个位置都唯一的位置id。这样，在一段时间内的同一个用户的签到地点就可以构成一段位置序列数据。我们可以通过下面的方式加载原始的Gowalla数据：

```{code-cell} ipython3
from trajdl.datasets.open_source.conf import GowallaDataset

# 加载Gowalla数据集的元信息，打印元信息
gowalla = GowallaDataset()
print(gowalla)

# 以polars DataFrame的类型加载数据
df_gowalla = gowalla.load(return_as="pl")
df_gowalla.head()
```

通过运行上述命令可以看到TrajDL定义的公开数据集的信息，包含数据名、数据大小、下载链接，SHA-256等信息。

`return_as`参数可以控制`load`方法返回的数据类型，可以是`polars.DataFrame`（默认，适合高性能数据处理的场景）， `pandas.DataFrame`（适合于现有科学计算和数据分析工具兼容的场景），`pyarrow.Table`（适合高效的数据存储、读取和传输的场景）。三者之间可以通过简单的命令进行转换，用户根据自己的需求可自行选择返回的类型。

+++

## Porto数据集

[Porto](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)数据集收集了在葡萄牙波尔图市运行的442辆出租车的全年轨迹（2013年7月1日至2014年6月30日）。我们可以通过下面的方式加载原始的Porto数据集:

```{code-cell} ipython3
from trajdl.datasets.open_source.conf import PortoDataset

# 加载Porto数据集元信息并打印
porto = PortoDataset()
print(porto)

# 以polars.DataFrame的形式返回，展示前两行
df_porto = porto.load(return_as="pl")
df_porto.head(2)
```

该原始数据各字段的含义如下:
|列名|数据类型|含义|
|-------|------|-----|
|TRIP_ID|String|每次行程的唯一标识|
|CALL_TYPE|Char|标识此服务的方式，为如下三者之一:<br> &nbsp; 1. "A": 该行程从中心调度<br> &nbsp; 2. "B": 该行程要求特定站点的出租车<br> &nbsp; 3. "C": "其他"|
|ORIGIN_CALL|Integer|每个电话号码的唯一标识符，该电话号码至少请求过一次服务。如果CALL_TYPE=’A’，则表示行程的客户|
|ORIGIN_STAND|Integer|出租车站的唯一标识。如果CALL_TYPE="B"，则表示行程的起点|
|TAXI_ID|Integer|每个行程出租车司机的唯一标识|
|TIMESTAMP|Integer|Unix时间戳，以秒为单位，表示该行程的开始时间|
|DAYTYPE|char|该行程开始时的日期类型，为如下三者之一: <br> &nbsp; 1. "B": 该形成在假日或其他特殊日期开始<br> &nbsp; 2. "C": 该行程在类型B的前一天开始<br> &nbsp; 3. "C": 其他(正常日、工作日、周末)
|MISSING_DATE|Boolean|当形成采集到的GPS数据流完整时，该值为FALSE;否则为TRUE|
|POLYLINE|String|GPS坐标列表字符串(WGS84坐标系)，字符串的开头和结尾用"["和"]"标识|

+++

## 自定义数据集管理

为了避开部分区域的网络限制，`TrajDL`支持用户自己下载公开数据集后通过接口加载到`TrajDL`中。`TrajDL`提供了3种方法来实现这个需求，下面我们会以Gowalla数据集为例介绍这几种方法。

### 方法1，修改环境变量

用户可以自行寻找下载速度较快的网站，并获得原始数据集的下载链接，比如`https://xxxxxx/datasets/loc-gowalla_totalCheckins.txt.gz`。每个公开数据集的url都有一个环境变量可以设置，`GowallaDataset`的环境变量名是`GOWALLA_URL`，可以通过下面的命令设置环境变量：

```bash
export GOWALLA_URL=https://xxxxxx/datasets/loc-gowalla_totalCheckins.txt.gz
```

然后再运行`TrajDL`相关的代码，比如

```python
from trajdl.datasets.open_source.conf import GowallaDataset

ds = GowallaDataset()
ds.load(return_as="pd").head()
```

`TrajDL`会读取环境变量并自动设置`URL`为环境变量里面指定的`URL`。

### 方法2，修改数据集的默认url

和上面的方法差不多，只不过`URL`是配置在代码内部。

```python
from trajdl.datasets.open_source.conf import GowallaDataset

gowalla_url = "https://xxxxxx/datasets/loc-gowalla_totalCheckins.txt.gz"

ds = GowallaDataset()
ds.set_url(gowalla_url)
ds.load(return_as="pd").head()
```

### 方法3，用户自行下载数据集之后通过接口配置

用户先将数据集下载到任意一个目录，比如`datasets/loc-gowalla_totalCheckins.txt.gz`，然后通过下面的代码就可以加载数据了。

```python
from trajdl.datasets.open_source.conf import GowallaDataset

original_dataset_path = "datasets/loc-gowalla_totalCheckins.txt.gz"

ds = GowallaDataset()
ds.load(return_as="pd", original_dataset_path=original_dataset_path).head()
```

目前`TrajDL`支持的公开数据集的环境变量定义如下：
| Dataset | Env         |
|---------|-------------|
| Gowalla | GOWALLA_URL |
| Porto   | PORTO_URL   |

```{note}

1. 推荐用户使用前两种方式，这两种方式适合用户有自己的数据管理系统，比如用户所在实验室或者企业有可以提供HTTP服务的存储工具，用户可以通过`URL`快速拉取数据集。这种方式在基于`Docker`的运行环境里面，优势比较明显。
2. `TrajDL`的代码里面配置了公开数据集的默认URL和SHA-256的值，用户需要下载相同的数据集才能正常使用上述两种方法完成数据加载。
```

+++

## 公开数据集如何转换为LocSeq或Trajectory

`TrajDL`的基础数据结构是`LocSeq`和`Trajectory`，用户在加载了公开数据集后，自己会进行一些数据处理，当数据都被处理好后就可以转换为`LocSeq`或`Trajectory`了，供后续使用。

+++

### 将Gowalla转换为LocSeq和LocSeqDataset

```{code-cell} ipython3
# 我们要用Polars的API进行数据处理
import polars as pl

from trajdl.datasets import LocSeq
from trajdl.datasets.open_source.conf import GowallaDataset

df = GowallaDataset().load()
df.head()
```

```{code-cell} ipython3
# 将check_in的时间按天划分，1个用户1天的位置组成1条位置序列
df_locseqs = (
    df
    .with_columns(pl.col("check_in_time").dt.strftime("%Y%m%d").alias("ds"))
    .group_by("user_id", "ds")
    .agg(pl.col("loc_id").sort_by(pl.col("check_in_time")).alias("loc_seq"))
    .select("user_id", "ds", "loc_seq")
)
df_locseqs.head()
```

```{code-cell} ipython3
# 取其中的前10条序列，构造List[LocSeq]和LocSeqDataset

from trajdl.datasets import LocSeq, LocSeqDataset

# 遍历dataframe的前10行，将其转换为LocSeq，entity_id设定为用户id
locseqs = []
for user_id, _, loc_seq in df_locseqs.head(10).iter_rows():
    locseqs.append(LocSeq(seq=loc_seq, entity_id=user_id))
locseqs
```

然后将其转换为LocSeqDataset

```{code-cell} ipython3
ds = LocSeqDataset.init_from_loc_seqs(locseqs)
ds
```

```{code-cell} ipython3
ds.schema()
```

```{code-cell} ipython3
ds.seq
```

### 将Porto转换为Trajectory和TrajectoryDataset

`TrajectoryDataset`同理通过上述方法构建，我们以`Porto`数据集为例。

```{code-cell} ipython3
from trajdl.datasets.open_source.conf import PortoDataset

df = PortoDataset().load()
df.head(2)
```

```{code-cell} ipython3
from trajdl.datasets import Trajectory, TrajectoryDataset
```

```{code-cell} ipython3
trajs = []
for polyline in df.head(10)["POLYLINE"]:
    trajs.append(Trajectory(seq=polyline.to_numpy()))
trajs
```

```{code-cell} ipython3
ds = TrajectoryDataset.init_from_trajectories(trajs)
ds
```

```{code-cell} ipython3
ds.schema()
```

```{tip}

本文介绍了`TrajDL`里面管理的公开数据集，以及如何将公开数据集转换成`BaseSeq`与`BaseArrowDataset`。用户在使用自己的数据集的时候也可以通过`BaseSeq`实现`BaseArrowDataset`的轻松构建。

```
