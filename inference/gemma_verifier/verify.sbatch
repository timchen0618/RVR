#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=verify
#SBATCH --output=/scratch/dq2024/diverse_retriever/inference/gemma_verifier/out_files/2B.out
#SBATCH --gres=gpu:2
#SBATCH --account=torch_pr_152_courant
#SBATCH --export=ALL 

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 

export HF_TOKEN='${HF_TOKEN}';
export VLLM_ATTENTION_BACKEND=TORCH_SDPA;

python gemma_verifier.py
"