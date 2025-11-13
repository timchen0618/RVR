#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=bm25
#SBATCH --output=out_files/bm25
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9;

python bm25.py --data /scratch/dq2024/diverse_retriever/train_data_queries.json \
            --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
            --output_dir /scratch/dq2024/diverse_retriever/data \
            --outfile qampari_bm25_base_t100.jsonl \
            --n_docs 100
"


