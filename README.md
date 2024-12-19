<div align="center">
  <img src="./docs/_static/images/wide-logo.svg" width=300>
  <h3><b> A python toolkit for Trajectory Deep Learning. </b></h3>
</div>

---

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_License_2.0-purple)](./LICENSE)
[![Docs](https://img.shields.io/badge/Docs-TrajDL-blue)](https://trajdl.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.1.0+-green)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.3.3+-green)](https://lightning.ai/docs/pytorch/stable/)


</div>

`TrajDL`提供了轨迹数据挖掘领域中的多个SOTA深度学习模型的实现，为研究人员、工程师提供易用、高效、可靠的开发工具，可以快速开展实验和应用开发。TrajDL有几个关键特性：

- **基于Arrow，Pytorch和Lightning**

  `TrajDL`的数据部分构建在`Arrow`之上，模型部分构建在`Pytorch`之上，训练与验证流程构建在`Lightning`之上，充分结合各个框架工具的优势。

- **高效的工具**

  `TrajDL`提供了高效的工具，比如高效的`Dataset`，`Tokenizer`，`GridSystem`。出色的零拷贝特性可以显著降低数据的处理时间，节省内存使用。高效的`Tokenizer`和`GridSystem`可以随时转换数据，无需预先处理数据。

- **可扩展性**

  `TrajDL`高度模块化，不会约束用户的代码，用户可以随时从`TrajDL`里面取出自己需要使用的工具。`TrajDL`还打通了与`Polars`，`Pandas`，`PyArrow`等工具的接口，用户使用常用的科学计算工具处理后的数据可以轻松导入到`TrajDL`的数据体系。另外`TrajDL`同时支持API与配置文件两种方式开展实验与开发，尽可能提升用户体验。

- **包含SOTA模型的实验复现脚本**

  `TrajDL`提供了SOTA模型的复现脚本，用户可以通过脚本重现论文内的实验结果，部分场景下`TrajDL`具备比论文场景更优的效果。


# 文档 📕

简体中文文档参阅：[简体中文文档](https://trajdl.readthedocs.io/en/latest/)

*English documentation will be provided in subsequent versions.*

# Benchmark 🚀

[`scripts/benchmark`](scripts/benchmark)目录下存储了`TrajDL`提供的benchmark复现脚本，针对各个论文使用`TrajDL`进行了实验复现。

# License
本项目使用**Apache License 2.0**，详见[LICENSE](LICENSE)。
