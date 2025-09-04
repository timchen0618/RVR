#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=h200_courant
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=qw-b-embed
#SBATCH --gres=gpu:2
#SBATCH --array=0-15%8
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/qwen2/qampari_base_finetuned/out_files/base-embed-passages-%a.out
#SBATCH --wrap "hostname && sleep infinity"


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 
pip show transformers


python ../generate_passage_embeddings.py \
        --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_base_finetuned/checkpoint/best_model \
        --output_dir /scratch/dq2024/diverse_retriever/training/qwen2/qampari_base_finetuned  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 \
        --per_gpu_batch_size 128
"
