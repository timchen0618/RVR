#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=webqsp
#SBATCH --output=out_files/webqsp-qwen3.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/torch/diverse_retriever/training/qwen2/passage_retrieval.py \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --passages /scratch/dq2024/torch/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/torch/diverse_retriever/inference/qwen3/passage_embeddings/qampari_base/passages_*' \
    --data /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_webqsp.jsonl \
    --output_dir /scratch/dq2024/torch/diverse_retriever/data \
    --outfile webqsp_qwen3_t500.json \
    --training_mode "base" \
    --n_docs 500 \
    --question_key question_text
"


