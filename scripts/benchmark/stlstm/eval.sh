folder=$1

ckpt_filename=$(python -c "from trajdl.utils import find_best_checkpoint;print(find_best_checkpoint('$folder'))")

echo $ckpt_filename

python ../../main.py test \
    --config configs/gowalla_config.yaml \
    --trainer.logger=False \
    --trainer.default_root_dir=None \
    --ckpt_path $folder/$ckpt_filename