# Data

`TrajDL`提供了一系列用于表示位置序列和轨迹序列的工具，可以帮助用户快速开展实验和应用开发。

简单来讲，`TrajDL`提供了两种粒度的数据：**单条序列**和**多条序列**。前者包含**位置序列**与**轨迹序列**，后者包含**位置序列数据集**和**轨迹序列数据集**。

此外，`TrajDL`还具备公开数据集的管理工具，用户可以通过该工具快速下载公开数据集开展实验。

`TrajDL`提供的这一系列数据管理工具的核心宗旨是为了**加速整个实验**过程，从数据预处理、模型训练、模型验证，高效的数据管理工具可以让整个机器学习流程的速度变快。

![Sequence And Dataset](../_static/images/sequence_and_dataset.svg)

```{toctree}
:maxdepth: 1

sequence_and_dataset
open_source_dataset

```