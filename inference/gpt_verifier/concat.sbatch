#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=concat
#SBATCH --output=/scratch/dq2024/diverse_retriever/inference/gpt_verifier/out_files/concat.out
#SBATCH --account=torch_pr_152_courant

API_KEY=$1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 

python /scratch/dq2024/diverse_retriever/inference/gpt_verifier/concat.py \
  /scratch/dq2024/diverse_retriever/inference/gpt_verifier/all_golds/round4_all_golds.json \
  /scratch/dq2024/diverse_retriever/inference/gpt_verifier/round4_top6_concatenated.json \
  --top_k 6
"
