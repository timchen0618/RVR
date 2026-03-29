#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=eval
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/out_files/rrf.out
#SBATCH --account=torch_pr_152_courant
#SBATCH --gres=gpu:1


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9;

python eval/rerank.py \
    --input_file /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t200.json \
    --output_file /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_reranked.json \
    --top_k 100 \
    --batch_size 32
"
