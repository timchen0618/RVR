#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dr02-base-retrieve
#SBATCH --output=/scratch/dq2024/diverse_retriever/out_files/inf-0-3.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/inf_retriever/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_standard_finetuned_0-12/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_standard_finetuned_infly_0-12/passages_*' \
    --data /scratch/dq2024/diverse_retriever/inference/infly/base/input_files/qampari_base_finetuned_infly_0-12_concat_200.json \
    --output_dir /scratch/dq2024/diverse_retriever/inference/infly/standard \
    --outfile qampari_std_finetuned_infly_0-12_t200.json \
    --training_mode "standard" \
    --n_docs 100 \
    --norm_query
"


