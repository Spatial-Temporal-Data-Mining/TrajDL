# TULER

对于TULER算法，需要先从原作者的仓库里面下载数据集，放到如下目录：
```
processed_data/Gowalla/gowalla_scopus_1104.dat
processed_data/Brightkite/brightkite_scopus_tra_1215.dat
```

因为作者没有提供数据集的具体处理代码，所以TrajDL无法复现作者处理后的数据，需要自行下载。

然后通过下面的命令进行实验复现。

```bash
# samples
time bash preprocessing.sh --dataset gowalla --data_path processed_data/Gowalla/gowalla_scopus_1104.dat

# pretrain
# time bash pretrain.sh --ds output/gowalla/train_ds.parquet --output output/gowalla/word2vec.model
time bash pretrain.sh --ds output/gowalla/full_ds.parquet --output output/gowalla/word2vec.model

# train
time bash train.sh
```

1. 预处理的数据都会存放到`output/gowalla`目录下
2. 预训练的时候如果--ds指定的是`full_ds.parquet`，那么会用161万条数据训练word2vec，`train_ds.parquet`就是用训练集的序列训练。作者的论文代码是前者，即使用full_ds训练了word2vec模型。训练好的模型会存储到`output/gowalla`目录下
3. 训练脚本在每轮训练结束后，在测试集上评估了论文里面的三个指标。由于论文没有使用验证集，这里也不提供验证集的实现了。