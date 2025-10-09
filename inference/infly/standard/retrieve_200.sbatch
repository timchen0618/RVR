#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dr02-base-retrieve
#SBATCH --output=retrieve.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_base_finetuned_3/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_base_finetuned_infly_3/passages_*' \
    --data   \
    --output_dir /scratch/dq2024/diverse_retriever/inference/infly/standard \
    --outfile qampari_std_finetuned_infly_t200.json \
    --training_mode "base" \
    --n_docs 200 \
    --norm_query
"


