#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=h200,l40s
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=qw-b-embed
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/qwen2/out_files/qampari_base/base-embed-passages.out
#SBATCH --wrap "hostname && sleep infinity"


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../generate_passage_embeddings.py \
        --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
        --output_dir /scratch/dq2024/diverse_retriever/training/qwen2/qampari_base  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
"