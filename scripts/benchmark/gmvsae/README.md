# GMVSAE

```bash
# samples
time bash preprocessing.sh

# pretrain
time bash pretrain.sh 

# train
time bash train.sh --model.pretrain_ckpt_folder "output/porto/pretrain/lightning_logs/version_0/checkpoints/"

# evaluate
time bash evaluate.sh --ckpt_folder "output/porto/train/lightning_logs/version_0/checkpoints/"
```

1. 预处理的数据集会存放在`output/porto`目录下
2. 预训练的模型会存储在`output/porto/pretrain`目录下
3. 训练的模型会存储在`output/porto/train`目录下