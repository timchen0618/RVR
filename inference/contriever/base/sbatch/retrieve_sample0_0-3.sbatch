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
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/contriever/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_base_finetuned_only_random_sample_0/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/contriever/passage_embeddings/qampari_contriever_base_finetuned_only_random_sample_0/passages_*' \
    --data /scratch/dq2024/diverse_retriever/inference/contriever/base/base_varying_input/qampari_base_finetuned_only_ran_gold_AR_0-3_t200_concat.json \
    --output_dir /scratch/dq2024/diverse_retriever/inference/contriever/base \
    --outfile qampari_corpus_finetuned_base_sample_0_t200_0-3.json \
    --n_docs 200
"