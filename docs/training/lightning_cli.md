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

# LightningCLI

```{attention}

在1.0.0版本发布前，当前文档的内容可能会发生变化。

```

`TrajDL`是构建在`Pytorch Lightning`上的算法工具包，因此其设计的每个模型都支持`LightningCLI`，而且其提供的`BaseSeqDataModule`是基于`LightningModule`构建的，其子类都支持`LightningModule`的特性，完全适配`LightningCLI`。用户可以通过配置文件的方式快速拉起训练任务，这在批量训练模型的场景里面非常实用。尤其是在实验和生产环境里面。


```{seealso}
用户可以查询[LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)的官方文档了解配置文件的语法。
```

`TrajDL`针对每个SOTA算法提供了`YAML`配置文件，用户可以通过`TrajDL`提供的脚本快速运行SOTA模型的实验，所有脚本放置在源代码仓库内的`scripts/`目录下。

```{code-cell} ipython3

```
