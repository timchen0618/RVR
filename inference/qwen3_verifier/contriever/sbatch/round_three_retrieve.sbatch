#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ret0123
#SBATCH --output=out_files/r3.out
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
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_standard_only_random_0123/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/contriever/passage_embeddings/qampari_contriever_finetuned_standard_only_random_0123/passages_*' \
    --data /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/input_files/qampari_finetuned_base_t200_promote-qwen3_0-3_concat_round_two.json \
    --output_dir /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever \
    --outfile qampari_std_t200_promote-qwen3_0-3_retrieved_round3_100.json \
    --training_mode "standard" \
    --n_docs 100
"
