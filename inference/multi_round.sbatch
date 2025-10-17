#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=multi-round-pipeline
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/contriever/multi_round_pipeline/out_files/pipeline.out

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/diverse_retriever/inference/multi_round_pipeline.py \
    --initial_merged_file /scratch/dq2024/diverse_retriever/eval/contriever/base/MERGED_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json \
    --ground_truth_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --passages_file /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/contriever/passage_embeddings/qampari_contriever_finetuned_standard_only_random_0123/passages_*' \
    --model_path /scratch/dq2024/diverse_retriever/models/diverse_retriever/checkpoints/qampari_contriever_standard_only_random_0123/checkpoint/best_model \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/contriever/multi_round_pipeline \
    --retrieval_script /scratch/dq2024/diverse_retriever/training/contriever/passage_retrieval.py \
    --total_rounds 3 \
    --top_k_concat 3 \
    --n_docs_retrieve 200 \
    --training_mode base \
    --conda_env dr-env-3-9 \
    --norm_query
"