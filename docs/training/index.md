# Training

本章主要讲解训练过程需要接触的概念与内容，主要包含：
1. `collate_function`：在`DataLoader`里面操作`BaseArrowDataset`。
2. `BaseSeqDataModule`：在`Lightning`的体系里面，如果使用`Lightning Trainer`进行训练，需要用户编写`LightningDataModule`，`TrajDL`提供了封装好的`BaseSeqDataModule`。
3. `LightningCLI`：`TrajDL`支持使用`LightningCLI`进行模型训练与验证。

```{toctree}
:maxdepth: 1

collate_function
lightning_data_module
lightning_cli

```