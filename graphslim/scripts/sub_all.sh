#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=a100v100
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a100-8-gm320-c96-m1152
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate pygdgl
cd ~/GraphSlim/graphslim

echo '====start running===='
python train_all.py -M doscondx --save_path /scratch/sgong36/checkpoints --load_path /scratch/sgong36/data --epochs 10 --init clustering --lr_adj 1e-4 --lr_feat 1e-4
echo '=====end======='