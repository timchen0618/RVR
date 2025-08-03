#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=h200, l40s
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=lprj-embed-0123
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%1
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/contriever/qampari_finetuned_lin_proj_only_random_sample_0123/out_files/embed-passages-%a.out
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../generate_passage_embeddings.py \
        --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_lin_proj_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48_pos1_sample_0123/checkpoint/best_model \
        --output_dir /scratch/dq2024/diverse_retriever/training/contriever/qampari_finetuned_lin_proj_only_random_sample_0123  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 
"