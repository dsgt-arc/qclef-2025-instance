#!/bin/bash
#SBATCH -J bcos                      # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH -N1 --gres=gpu:V100:1             # Number of GPU
#SBATCH --cpus-per-task=8                       # Number of CPU
##SBATCH --mem-per-gpu=16G                       # Memory of GPU
#SBATCH -t0:120:00                              # Duration
#SBATCH -qembers                                # QOS Name (Jobs finishing in less than an hour -> ember)
#SBATCH --output=Report-%j.log                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=apramov3@gatech.edu         # E-mail address for notifications
set -xeu

module load python/3.10
 
 
source /storage/home/hcoda1/9/apramov3/scratch/qclef-2025-instance/.venv/bin/activate
 
cd ~/storage/home/hcoda1/9/apramov3/github/qclef-2025-instance

python bcos_run.py

