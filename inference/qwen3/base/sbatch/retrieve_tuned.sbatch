#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=qwen-500
#SBATCH --output=out_files/base-retrieve.out
#SBATCH --gres=gpu:1


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/diverse_retriever/training/qwen2/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_base_finetuned_10k_query+keywords/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/qwen3/passage_embeddings/qampari_qwen3_base_finetuned_10k_query+keywords/passages_*' \
    --data /scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qampari_qwen3_original+keywords.json \
    --output_dir /scratch/dq2024/diverse_retriever/training/query_rewrite/qwen3 \
    --outfile qampari_qwen3_finetuned_10k_base_original+keywords_retrieved.json \
    --training_mode "base" \
    --n_docs 100 \
    --norm_query
"
