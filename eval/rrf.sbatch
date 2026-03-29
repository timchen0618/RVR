#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=eval
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/out_files/rrf.out
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9;

python /scratch/dq2024/diverse_retriever/eval/rrf.py \
    /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/retrieved_files/round2_retrieved.json \
    /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/retrieved_files/round2_retrieved.json \
    /scratch/dq2024/diverse_retriever/eval/contriever+infly_rrf_res_round2.json
"
