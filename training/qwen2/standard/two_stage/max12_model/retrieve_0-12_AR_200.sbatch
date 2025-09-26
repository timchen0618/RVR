#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=0-12-retr
#SBATCH --output=out_files/0-12.out
#SBATCH --gres=gpu:1


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python ../../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_standard_finetuned_only_random_0-12/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/qwen2/qampari_standard_finetuned_0-12/passages_*' \
    --data /scratch/dq2024/diverse_retriever/training/qwen2/base/two_stage/base_top0-12/qampari_standard_only_rand_gold_AR_0-12_concat_200.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/qwen2/standard/two_stage \
    --outfile qampari_corpus_base_r1qw_r2std_AR_0-12_t200.json \
    --training_mode "standard" \
    --n_docs 100 \
    --norm_query
"
