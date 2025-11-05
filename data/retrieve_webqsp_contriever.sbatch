#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=webqsp
#SBATCH --output=out_files/webqsp-contr.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/torch/diverse_retriever/training/contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages /scratch/dq2024/torch/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/torch/diverse_retriever/inference/contriever/passage_embeddings/qampari_base_only_random/passages_*' \
    --data /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_webqsp.jsonl \
    --output_dir /scratch/dq2024/torch/diverse_retriever/data \
    --outfile webqsp_contreiver_t500.json \
    --training_mode "base" \
    --n_docs 500 \
    --question_key question_text
"


