#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=verify
#SBATCH --output=/scratch/dq2024/diverse_retriever/inference/gpt_verifier/out_files/verifyq.out
#SBATCH --account=torch_pr_152_courant

API_KEY=$1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 

python /scratch/dq2024/diverse_retriever/inference/gpt_verifier/gpt5_verifier.py --api-key $API_KEY
"
