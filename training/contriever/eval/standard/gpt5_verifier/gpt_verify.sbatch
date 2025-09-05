#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=verify
#SBATCH --output=verify.out
#SBATCH --wrap "hostname && sleep infinity"

API_KEY=$1


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python gpt5_verifier.py --api-key $API_KEY 
"
