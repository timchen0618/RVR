#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=embed
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16
#SBATCH --output=out_files/base-embed-passages-%a.out
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/inf_retriever/generate_passage_embeddings.py \
        --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_standard_only_random_0-6_retrieved_docs/checkpoint/best_model \
        --output_dir /scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_infly_standard_only_random_0-6_retrieved_docs \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 \
        --per_gpu_batch_size 64 \
        --norm_doc
"