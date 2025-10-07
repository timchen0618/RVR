#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=dr-base-embed
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%1
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/contriever/qampari_base_only_random/out_files/base-embed-passages-%a.out

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../generate_passage_embeddings.py \
        --model_name_or_path facebook/contriever-msmarco \
        --output_dir /scratch/dq2024/diverse_retriever/training/contriever/qampari_base_only_random  \
        --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
        --shard_id \$SLURM_ARRAY_TASK_ID \
        --num_shards 16 
"