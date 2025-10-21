#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dr02-base-retrieve
#SBATCH --output=retrieve.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/training/inf_retriever/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_base_finetuned_t0.05_lr0.00001_ws1000_bs48_acc2_gradchkpt/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_base_finetuned_infly_2/passages_*' \
    --data /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/input_files/qampari_finetuned_base_t200_promote-qwen3_0-6_concat.json \
    --output_dir /scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier \
    --outfile qampari_base_finetuned_infly_0-6_q3verif_input_test.json \
    --training_mode "base" \
    --n_docs 100 \
    --norm_query
"


