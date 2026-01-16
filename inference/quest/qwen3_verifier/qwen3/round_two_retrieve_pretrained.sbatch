#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=qwb
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/qwen3/pretrained/out_files/qwen3-base.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/diverse_retriever/training/qwen2/passage_retrieval.py \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/qwen3/passage_embeddings/qampari_base/passages_*' \
    --data /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/pretrained/base/verified_input/round1_top6_concatenated.json \
    --output_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/qwen3/pretrained/retrieved_files \
    --outfile round2_retrieved.json \
    --n_docs 100 \
    --norm_query
"
