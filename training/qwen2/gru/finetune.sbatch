#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=qwen-gru-tune
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=out_files/gru/gru-finetune.out
#SBATCH --gres=gpu:4

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

accelerate launch --main_process_port 29500 ../finetuning.py --train_data ..././../train_random+input_data.jsonl  \
                                                 --eval_data ../../../train_eval_random+input_data.jsonl  \
                                                 --temperature 0.05 \
                                                 --total_steps 50000 \
                                                 --warmup_steps 1000 \
                                                 --lr 0.0001 \
                                                 --save_freq 2500 \
                                                 --log_freq 50 \
                                                 --eval_freq 2500 \
                                                 --negative_hard_ratio 0.0 \
                                                 --negative_ctxs 1 \
                                                 "" \
                                                 --per_gpu_batch_size 48 \
                                                 --per_gpu_eval_batch_size 48 \
                                                 --model_path Qwen/Qwen2-0.5B-Instruct \
                                                 --chunk_length 512 \
                                                 --accumulation_steps 2 \
                                                 --run_name "qampari_qwen_finetuned_gru_only_random_0123" \
                                                 --output_dir "/scratch/dq2024/models/diverse_retriever/checkpoints/" \
                                                 --training_mode "gru" \
                                                 --max_positive_documents 1 \
                                                 --sample_length \
                                                 --doc_length 0 1 2 3
"