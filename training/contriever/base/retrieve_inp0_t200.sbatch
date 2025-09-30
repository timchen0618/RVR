#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ret
#SBATCH --output=base-rewrite.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_base_finetuned_only_random/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/contriever/qampari_base_finetuned_only_random/passages_*' \
    --data /scratch/dq2024/diverse_retriever/training/query_rewrite/qampari_contriever_rewritten_queries_expansion.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/base \
    --outfile qampari_corpus_base_query_rewrite.json \
    --training_mode "base" \
    --n_docs 100
"


