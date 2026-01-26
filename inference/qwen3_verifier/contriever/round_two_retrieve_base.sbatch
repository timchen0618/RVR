#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=rcbv100
#SBATCH --output=/scratch/dq2024/diverse_retriever/query_rewrite/contriever/ontr-base.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/contriever/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_base_finetuned_only_random_sample_0/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/contriever/passage_embeddings/qampari_contriever_base_finetuned_only_random_sample_0/passages_*' \
    --data /scratch/dq2024/diverse_retriever/query_rewrite/contriever/contriever_query+keywords_max3.json \
    --output_dir /scratch/dq2024/diverse_retriever/query_rewrite/contriever \
    --outfile contriever_query+keywords_max3_retrieved.json \
    --training_mode "standard" \
    --n_docs 100 \
    --save_or_load_index
"
