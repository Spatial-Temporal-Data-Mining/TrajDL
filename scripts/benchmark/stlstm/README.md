# STLSTM

```bash
# samples
time bash preprocessing.sh

# pretrain
time bash pretrain.sh --ds "output/gowalla/train_ds.parquet" --output "output/gowalla/word2vec.model"

# train
time bash train.sh

# evaluate
time bash eval.sh "output/gowalla/lightning_logs/version_0/checkpoints/"
```

1. 预处理的数据会存放在`output/gowalla`目录下
2. 预训练的word2vec模型会存放在`output/gowalla`目录下
3. 训练的模型会存放在`output/gowalla/lightning_logs`目录下