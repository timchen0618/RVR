#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=inf
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100itr/infly/out_files/infly.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/inf_retriever/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_standard_finetuned_0-6/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_standard_finetuned_infly_0-6/passages_*' \
    --data /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100itr/infly/verified_input/round1_top6_concatenated.json \
    --output_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100itr/infly/retrieved_files \
    --outfile round2_retrieved.json \
    --training_mode "standard" \
    --n_docs 100 \
    --norm_query
"
