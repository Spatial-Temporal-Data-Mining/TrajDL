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

# Data Minipulation

+++

TrajDL对数据类型进行了抽象，支持位置序列和轨迹序列两种数据集。在数据集的来源上区分为私有数据集和公开数据集。

TrajDL提供了位置序列`LocSeq`和轨迹`Trajectory`两种类型，用户需要自行将数据转换为两种类型组成的`List`，然后将其放入`LocSeqDataset`和`TrajectoryDataset`便可完成数据的管理和使用。

+++

## 公开数据集管理

公开数据集接口可以帮助管理公开数据集，如Gowalla, porto等。用户可以通过这类接口快速下载原始数据集并轻松构建缓存，提升数据集的加载效率。

我们以Gowalla为例，介绍如何使用公开数据集的接口。

```{code-cell} ipython3
from trajdl.datasets.open_source import GowallaDataset

# 加载Gowalla数据集的元信息，打印元信息
gowalla = GowallaDataset()
print(gowalla)
```

通过运行上述命令可以看到TrajDL定义的公开数据集的信息，包含数据名、数据大小、下载链接，SHA-256等信息。

```{code-cell} ipython3
# 以polars DataFrame的类型加载数据
df = gowalla.load(return_as="pl")
df.head()
```

除了polars.DataFrame, TrajDL还提供了pandas.DataFrame, pyarrow.Table类型的支持。

```{code-cell} ipython3
# 以pandas.DataFrame的形式返回
df = gowalla.load(return_as="pd")
df.head()
```

```{code-cell} ipython3
# 以pyarrow.Table的形式返回
table = gowalla.load(return_as="pa")
table.schema
```

公开数据集的格式与其官方提供的定义是一致的，用户可以通过上述接口自行处理数据。

+++

### 自定义数据集管理

为了避开部分区域的网络限制，TrajDL支持用户自己下载公开数据集后通过接口加载到TrajDL中。TrajDL提供了3种方法来实现这个需求，下面我们会以Gowalla数据集为例介绍这几种方法。

#### 方法1，修改环境变量

用户可以自行寻找下载速度较快的网站，并获得原始数据集的下载链接，比如`https://xxxxxx/datasets/loc-gowalla_totalCheckins.txt.gz`。每个公开数据集的url都有一个环境变量可以设置，GowallaDataset的环境变量名是GOWALLA_URL，可以通过下面的命令设置环境变量：

```bash
export GOWALLA_URL=https://xxxxxx/datasets/loc-gowalla_totalCheckins.txt.gz
```

然后再运行TrajDL相关的代码，比如

```python
from trajdl.datasets.open_source.conf import GowallaDataset

ds = GowallaDataset()
ds.load(return_as="pd").head()
```

TrajDL会读取环境变量并自动设置url为环境变量里面指定的url。

#### 方法2，修改数据集的默认url

和上面的方法差不多，只不过url是配置在代码内部。

```python
from trajdl.datasets.open_source.conf import GowallaDataset

gowalla_url = "https://xxxxxx/datasets/loc-gowalla_totalCheckins.txt.gz"

ds = GowallaDataset()
ds.set_url(gowalla_url)
ds.load(return_as="pd").head()
```

#### 方法3，用户自行下载数据集之后通过接口配置

用户先将数据集下载到任意一个目录，比如`datasets/loc-gowalla_totalCheckins.txt.gz`，然后通过下面的代码就可以加载数据了。

```python
from trajdl.datasets.open_source.conf import GowallaDataset

original_dataset_path = "datasets/loc-gowalla_totalCheckins.txt.gz"

ds = GowallaDataset()
ds.load(return_as="pd", original_dataset_path=original_dataset_path).head()
```

#### 备注

TrajDL的代码里面配置了公开数据集的默认url和SHA-256的值，用户需要下载相同的数据集才能正常使用上述两种方法完成数据加载。

```{code-cell} ipython3

```
