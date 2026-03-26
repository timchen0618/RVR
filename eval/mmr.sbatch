#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/out_files/qwen_mmr.out
#SBATCH --account=torch_pr_152_courant
 
 
if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi
 
 
singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh;
conda activate dr-env-3-9;
python /scratch/dq2024/diverse_retriever/eval/mmr.py \
    --input /scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_corpus_finetuned_base_t200.json \
    --output /scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_corpus_finetuned_base_t100_mmr_from_200.json \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_qwen3_base_finetuned_t0.05_lr0.00001_ws1000_bs48_acc2_rightpad_gradchkpt/checkpoint/best_model \
    --model_type infly \
    --top_k 100 \
    --lambda_param 0.5 \
    --per_gpu_batch_size 64 \
    --question_maxlength 512 \
    --training_mode "base"
"
 