#!/bin/bash
# #SBATCH --account students
#SBATCH --partition cpu
#SBATCH -N 1 # number of nodes
# #SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 1-00:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"
#SBATCH --mem=8G # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed
#SBATCH --qos=cpu # this qos ensures a very high priority but only one job per user can run under this mode.
#SBATCH --cpus-per-task=1

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr

tensorboard --logdir /nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/results/ResNet18
#tensorboard --logdir /nfs/students/ayle/guided-research/ResNet18-week5