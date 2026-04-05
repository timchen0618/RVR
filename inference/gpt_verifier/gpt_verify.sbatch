#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=verify
#SBATCH --output=/scratch/dq2024/diverse_retriever/inference/gpt_verifier/out_files/verifyq2.out
#SBATCH --account=torch_pr_152_courant

API_KEY=$1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 

python gpt_verifier_parallel.py \
    --api-key $API_KEY \
    --input-file /scratch/dq2024/diverse_retriever/inference/gpt_verifier/round2_retrieved.json \
    --output-file /scratch/dq2024/diverse_retriever/inference/gpt_verifier/all_golds/round2_all_golds.json \
    --batch-size 5 \
    --max-concurrent 50 \
    --parallel-queries 5
"