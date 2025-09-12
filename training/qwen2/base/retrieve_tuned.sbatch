#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dr-qbase-retrieve
#SBATCH --output=out_files/base-retrieve.out
#SBATCH --gres=gpu:1


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python ../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_base_finetuned_t0.05_lr0.00001_ws1000_bs48_acc2_rightpad_gradchkpt/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/qwen2/qampari_base_finetuned/passages_*' \
    --data /scratch/dq2024/diverse_retriever/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp0.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/qwen2/base \
    --outfile qampari_corpus_finetuned_base_t100.json \
    --training_mode "base" \
    --n_docs 100 \
    --norm_query
"
