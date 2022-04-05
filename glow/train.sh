#!/bin/bash
# #SBATCH --account students
#SBATCH --partition gpu_all
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 2-00:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"
#SBATCH --mem=32G # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed
# #SBATCH --qos=interactive # this qos ensures a very high priority but only one job per user can run under this mode.
#SBATCH --cpus-per-task=1

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr

#python train.py /nfs/students/ayle/guided-research/FASHION-jpg/training --img_size 32 --channels 3 --prune_criterion Johnit --pruning_limit 0.7 --checkpoint checkpoint/model_042001.pt

#python train.py /nfs/students/ayle/guided-research/FASHION-jpg/training --img_size 32 --channels 3 --batch 64 --prune_criterion Johnit --pruning_limit 0.9 --local_pruning --checkpoint checkpoint/model_criterion=EmptyCrit_sparsity=0.0_local=False_stable_011251.pt

#python train.py /nfs/students/ayle/guided-research/CIFAR-10-images/train --img_size 32 --channels 3 --batch 64 --prune_criterion Johnit --pruning_limit 0.7 --local_pruning --checkpoint checkpoint/model_dataset=train_criterion=EmptyCrit_sparsity=0.0_local=False.pt  # --optim_checkpoint checkpoint/optim_dataset=train_criterion=EmptyCrit_sparsity=0.0_local=False.pt

#python train.py CELEBA --img_size 32 --channels 3 --batch 64 --prune_criterion EmptyCrit  --checkpoint checkpoint/model_dataset=CELEBA_criterion=EmptyCrit_sparsity=0.0_local=False.pt --optim_checkpoint checkpoint/optim_dataset=CELEBA_criterion=EmptyCrit_sparsity=0.0_local=False.pt  # --pruning_limit 0.7 --local_pruning --checkpoint checkpoint/model_dataset=train_criterion=EmptyCrit_sparsity=0.0_local=False.pt  # --optim_checkpoint checkpoint/optim_dataset=train_criterion=EmptyCrit_sparsity=0.0_local=False.pt

python train.py CELEBA --img_size 32 --channels 3 --batch 64 --prune_criterion SNIPit --pruning_limit 0.5 --local_pruning --checkpoint checkpoint/model_dataset=CELEBA_criterion=EmptyCrit_sparsity=0.0_local=False.pt  # --optim_checkpoint checkpoint/optim_dataset=train_criterion=EmptyCrit_sparsity=0.0_local=False.pt
