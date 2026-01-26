#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ctr-multi
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v50/contriever/out_files/pipeline.out
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
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json \
    --ground_truth_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --passages_file /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/contriever/passage_embeddings/qampari_contriever_finetuned_standard_only_random_0123/passages_*' \
    --model_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_standard_only_random_0123/checkpoint/best_model \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v50/contriever \
    --retrieval_script /scratch/dq2024/diverse_retriever/training/contriever/passage_retrieval.py \
    --total_rounds 2 \
    --top_k_concat 3 \
    --n_docs_retrieve 100 \
    --training_mode standard \
    --conda_env dr-env-3-9 \
    --topk 50
"