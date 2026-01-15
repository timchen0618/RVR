#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=contrb
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/contriever/pretrained/out_files/contr-base.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/contriever/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/contriever/passage_embeddings/qampari_base_only_random/passages_*' \
    --data /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/contriever/pretrained/verified_input/round1_top3_concatenated.json \
    --output_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/quest/v50/contriever/pretrained/retrieved_files \
    --outfile round2_retrieved.json \
    --training_mode "base" \
    --n_docs 100
"
