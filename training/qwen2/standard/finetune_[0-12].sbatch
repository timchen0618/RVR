#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=std-tune
#SBATCH --output=out_files/std-tune-0-12.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

accelerate launch --main_process_port 29501 ../finetuning.py --train_data ../../../train_data_10k_query+keywords.jsonl \
                                                 --eval_data ../../../train_eval_data.jsonl \
                                                 --temperature 0.05 \
                                                 --total_steps 50000 \
                                                 --warmup_steps 1000 \
                                                 --lr 0.00001 \
                                                 --save_freq 2500 \
                                                 --log_freq 25 \
                                                 --eval_freq 2500 \
                                                 --negative_hard_ratio 0.0 \
                                                 --negative_ctxs 1 \
                                                 --per_gpu_batch_size 48 \
                                                 --per_gpu_eval_batch_size 48 \
                                                 --model_path Qwen/Qwen3-Embedding-0.6B \
                                                 --chunk_length 512 \
                                                 --accumulation_steps 2 \
                                                 --run_name "qampari_qwen3_standard_finetuned_only_random_0-12_10k_query+keywords" \
                                                 --output_dir "/scratch/dq2024/models/diverse_retriever/checkpoints/" \
                                                 --training_mode "standard" \
                                                 --max_positive_documents 1 \
                                                 --sample_length \
                                                 --doc_length 0 1 2 3 4 5 6 7 8 9 10 11 12 \
                                                 --norm_doc \
                                                 --norm_query
"
