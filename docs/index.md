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

# TrajDL - 基于深度学习的轨迹序列算法包

---

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

*English documentation will be provided in subsequent versions.*

```

+++

`TrajDL`提供了轨迹数据挖掘领域中的多个SOTA深度学习模型的实现，为研究人员、工程师提供易用、高效、可靠的开发工具，可以快速开展实验和应用开发。

基于Arrow，Pytorch和Lightning
: `TrajDL`的数据部分构建在`Arrow`之上，模型部分构建在`Pytorch`之上，训练与验证流程构建在`Lightning`之上，充分结合各个框架工具的优势。

高效的工具
: `TrajDL`提供了高效的工具，比如高效的`Dataset`，`Tokenizer`，`GridSystem`。出色的零拷贝特性可以显著降低数据的处理时间，节省内存使用。高效的`Tokenizer`和`GridSystem`可以随时转换数据，无需预先处理数据。

可扩展性
: `TrajDL`高度模块化，不会约束用户的代码，用户可以随时从`TrajDL`里面取出自己需要使用的工具。`TrajDL`还打通了与`Polars`，`Pandas`，`PyArrow`等工具的接口，用户使用常用的科学计算工具处理后的数据可以轻松导入到`TrajDL`的数据体系。另外`TrajDL`同时支持API与配置文件两种方式开展实验与开发，尽可能提升用户体验。

SOTA模型的实验复现脚本
: `TrajDL`提供了SOTA模型的复现脚本，用户可以通过脚本重现论文内的实验结果，部分场景下`TrajDL`具备比论文场景更优的效果。

---

+++

::::{grid} 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` 高效的数据管理
:link: data/index
:link-type: doc

`TrajDL`提供了**公开数据集**的管理工具，用户可以轻松下载公开数据集开展自己的实验，内置的缓存功能可以显著加快实验速度。

同时`TrajDL`还提供了基于`Arrow`的**高效数据管理**工具，可以帮助用户从不同粒度实现序列数据的管理，其**出色的扩展性**支持用户直接导入`Polars`，`Pandas`，`PyArrow`处理后的数据，训练过程中也不会因为多进程导致数据的拷贝复制，**显著节省内存**。

+++

[Learn more »](data/index)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` 易用的Tokenizer与网格系统
:link: tools/index
:link-type: doc

`TrajDL`提供了高效的`Tokenizer`与网格系统`GridSystem`。`Tokenizer`可以快速完成**Vocabulary的构建**，**Special Tokens的管理**，**快速的位置映射**。`GridSystem`可以将区域划分成网格，将经纬度转换成位置id。`Tokenizer`与`GridSystem`都有**多种类型**可以选择。

+++

[Learn more »](tools/index)
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` 高效简单的训练验证pipeline
:link: training/index
:link-type: doc

基于`Lightning`构建的`TrajDL`可以**快速开展**轨迹序列数据上深度学习的实验，`Lightning Trainer`，`Lightning Fabric`两种方式用户可以**自行选择**，通过API控制模型的训练与评估。`LightningCLI`让训练验证通过配置文件快速启动。`TrajDL`提供了封装好的`DataModule`和SOTA模型，用户只要**几行代码**就可以快速拉起实验。

+++

[Learn more »](training/index)
:::

::::

+++

```{toctree}
:caption: 🚀 Quick Start
:hidden: true
:maxdepth: 2
:name: Quick Start

getting_started/installation
getting_started/quick_start
```

```{toctree}
:caption: 📚 Tutorials
:hidden: true
:maxdepth: 2
:name: Tutorials

data/index
tools/index
training/index
advanced/index
```

```{toctree}
:caption: 📜 Algorithms
:hidden: true
:maxdepth: 2
:name: Algorithms

available_algorithms/index
examples/index
```

```{toctree}
:caption: 📔 Supplementary
:hidden: true
:maxdepth: 2
:name: Supplementary

supplementary/benchmarking
supplementary/references
```

```{toctree}
:caption: 📏 API Docs
:hidden: true
:maxdepth: 2
:name: API Docs

api/trajdl/trajdl
```
