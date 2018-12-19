#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mo-n3c
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_naive.txt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -t 72:00:00
#SBATCH --mem 32G
# sends mail when process begins, and
# when it ends. Make sure you difine your email
# address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=runzhey@cs.princeton.edu

export PATH="/home/runzhey/miniconda3/bin:$PATH"
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
source activate rllab3

cd /home/runzhey/MORL/multimario/
sh scripts/train_naive.sh
