# t2vec

```bash
# samples
time bash preprocessing.sh

# train
time bash train.sh

# evaluate
time bash eval.sh "output/porto" "output/porto/lightning_logs/version_0/checkpoints/model-epoch=006-val_loss=14.193707.ckpt"
```

1. 预处理会构建训练、验证、测试集，tokenizer存储在`output`目录下
2. 模型训练的checkpoint和tensorboard日志会存储在`output/lightning_logs`目录下
3. 评测时生成的内容会存储到`output/porto/eval`目录下