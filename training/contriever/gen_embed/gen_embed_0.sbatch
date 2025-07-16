#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=dr-gru-embed
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/contriever/qampari_finetuned_gru_only_random/out_files/gru-embed-passages-0.out

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python generate_passage_embeddings.py \
        --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_gru_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48_pos1_2gpu/checkpoint/best_model \
        --output_dir /scratch/dq2024/diverse_retriever/training/contriever/qampari_finetuned_gru_only_random  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id 0 \
        --num_shards 16 
"