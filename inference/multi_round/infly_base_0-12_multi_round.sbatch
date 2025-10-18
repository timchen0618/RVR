#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=inf-multi
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/infly/max12/base/out_files/pipeline.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/diverse_retriever/inference/multi_round_pipeline.py \
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t200.json \
    --ground_truth_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --passages_file /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_base_finetuned_infly_2/passages_*' \
    --model_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_base_finetuned_t0.05_lr0.00001_ws1000_bs48_acc2_gradchkpt/checkpoint/best_model \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/infly/max12/base \
    --retrieval_script /scratch/dq2024/diverse_retriever/training/inf_retriever/passage_retrieval.py \
    --total_rounds 10 \
    --top_k_concat 12 \
    --n_docs_retrieve 200 \
    --training_mode standard \
    --conda_env dr-env-3-9-qwen \
    --norm_query
"