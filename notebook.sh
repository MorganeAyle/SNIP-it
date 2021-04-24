#!/bin/bash
# #SBATCH --account students
#SBATCH --partition gpu_all
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 0-01:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"
#SBATCH --mem=8G # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed
#SBATCH --qos=interactive # this qos ensures a very high priority but only one job per user can run under this mode.
#SBATCH --cpus-per-task=1

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr

# python3 main.py --model LeNet5RBF --data_set MNIST --prune_criterion EarlySNIP --pruning_limit 0.98 --epochs 80 --run_name _lenet5rbf_mnist_98

export XDG_RUNTIME_DIR="" # Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318
jupyter notebook --no-browser --ip=$(hostname).kdd.in.tum.de