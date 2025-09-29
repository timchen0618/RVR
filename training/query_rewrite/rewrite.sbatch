#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=rewrite
#SBATCH --output=rewrite.out
#SBATCH --gres=gpu:4

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate base;

python query_rewrite.py
"
