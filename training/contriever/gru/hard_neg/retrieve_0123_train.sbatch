#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=h200
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=gru-tr2-retrieve
#SBATCH --output=out_files/gru-ret-train-backup.out
#SBATCH --gres=gpu:1
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_gru_only_random_0123/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/contriever/qampari_contriever_finetuned_gru_only_random_0123/passages_*' \
    --data /scratch/dq2024/diverse_retriever/train_data.jsonl \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/gru/hard_neg \
    --outfile qampari_corpus_finetuned_gru_hard_neg_train_0123.json \
    --training_mode "gru" \
    --question_key question_text \
    --n_docs 100
"