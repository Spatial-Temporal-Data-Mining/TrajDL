folder=$1
ckpt_path=$2

echo $folder
echo $ckpt_path

python scripts/split_traj.py --exp exp1 --batch_size 1024 --folder $folder --ckpt_path $ckpt_path
python scripts/split_traj.py --exp exp2 --batch_size 1024 --folder $folder --ckpt_path $ckpt_path --rate 0.2 0.3 0.4 0.5 0.6
python scripts/split_traj.py --exp exp3 --batch_size 1024 --folder $folder --ckpt_path $ckpt_path --rate 0.2 0.3 0.4 0.5 0.6

python scripts/evaluate.py --folder $folder --exp exp1
python scripts/evaluate.py --folder $folder --exp exp2 --rate 0.2 0.3 0.4 0.5 0.6
python scripts/evaluate.py --folder $folder --exp exp3 --rate 0.2 0.3 0.4 0.5 0.6