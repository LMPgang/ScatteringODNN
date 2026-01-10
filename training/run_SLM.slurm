#!/bin/bash -l

#SBATCH --job-name=2lay_2k
#SBATCH --partition=CLUSTER
#SBATCH -N 1
#SBATCH --time=40-6:6:6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --mail-type=end
#SBATCH --nodelist=compute-0-2
#SBATCH --mail-user=liyuegang0816@sjtu.edu.cn
#SBATCH --output=outfiles/run_SLM.out
#SBATCH --error=errfiles/run_SLM.err

module load python/anaconda3

conda info --envs
conda activate yuegang

python run_SLM.py


