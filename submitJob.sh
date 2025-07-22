#!/bin/bash
#SBATCH -J SMBHs_PTA
#SBATCH --time=20:00:00
#SBATCH -A sghodla
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH -o ./Slurm_output/%A_%a.out

cd /datalake/sghodla/SMBH_PTA_final
export OMP_NUM_THREADS=16

mu_list=(8 8.7 9)
mu=${mu_list[$SLURM_ARRAY_TASK_ID]}
python sshRun.py "$mu"
