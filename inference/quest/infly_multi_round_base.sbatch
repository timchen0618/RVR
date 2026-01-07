#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ctr-wqsp
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/quest/multi_round_pipeline/v100/infly/base/out_files/pipeline.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python /scratch/dq2024/diverse_retriever/inference/multi_round_pipeline.py \
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/inference/quest/infly/quest_infly_t100.json \
    --ground_truth_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_quest.jsonl \
    --passages_file /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_base_finetuned_only_random/passages_*' \
    --model_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_base_finetuned_only_random/checkpoint/best_model \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/quest/multi_round_pipeline/v100/infly/base \
    --retrieval_script /scratch/dq2024/diverse_retriever/training/infly/passage_retrieval.py \
    --total_rounds 2 \
    --top_k_concat 6 \
    --n_docs_retrieve 100 \
    --training_mode base \
    --conda_env dr-env-3-9 \
    --topk 100
"