python ../../main.py fit \
    --model GMVSAE \
    --data GMVSAEDataModule \
    --config configs/base.yaml \
    --config configs/porto_train_config.yaml \
    "$@"