#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=h200,l40s
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dr-gru-embed
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%16
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/inf_retriever/qampari_base_infly/out_files/gru-embed-passages-%a.out
#SBATCH --wrap "hostname && sleep infinity"


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 


python generate_passage_embeddings.py \
        --model_name_or_path infly/inf-retriever-v1-1.5b \
        --output_dir /scratch/dq2024/diverse_retriever/training/inf_retriever/qampari_base_infly  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 
"