#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=qw-b-embed
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16
#SBATCH --constraint="l40s|h200"
#SBATCH --output=out_files/base-embed-passages-%a.out


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 
pip show transformers


python ../generate_passage_embeddings.py \
        --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
        --output_dir /scratch/dq2024/diverse_retriever/training/qwen2/qampari_base_finetuned  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 \
        --per_gpu_batch_size 128
"
