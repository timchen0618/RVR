#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=emb-retr
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16
#SBATCH --output=out_files/emb-retr-%a.out
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../generate_passage_embeddings.py \
        --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_standard_only_random_123_both_objectives/checkpoint/best_model \
        --output_dir /scratch/dq2024/diverse_retriever/training/contriever/qampari_contriever_standard_only_random_123_both_objectives \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 
"
