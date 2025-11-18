#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=qw-b-embed
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16
#SBATCH --output=/scratch/dq2024/diverse_retriever/inference/qwen3/base/sbatch/out_files/base-tuned-embed-passages-%a.out
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 
pip show transformers


python /scratch/dq2024/diverse_retriever/training/qwen2/generate_passage_embeddings.py \
        --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_standard_finetuned_only_random_0-6-lr0.00005-temp0.05-steps25k/checkpoint/best_model \
        --output_dir /scratch/dq2024/diverse_retriever/inference/qwen3/passage_embeddings/qampari_qwen3_base_finetuned_0-6-lr0.00005-temp0.05-steps25k \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 \
        --per_gpu_batch_size 64 \
        --norm_doc
"
