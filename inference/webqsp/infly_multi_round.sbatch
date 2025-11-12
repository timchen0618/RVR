#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=inf-wqsp
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/infly/out_files/pipeline.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/diverse_retriever/inference/multi_round_pipeline_checkpt.py \
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/inference/webqsp/infly/webqsp_infly_t200.json \
    --ground_truth_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_webqsp.jsonl \
    --passages_file /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_base_infly/passages_*' \
    --model_path infly/inf-retriever-v1-1.5b \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/webqsp/multi_round_pipeline/infly \
    --retrieval_script /scratch/dq2024/diverse_retriever/training/inf_retriever/passage_retrieval.py \
    --total_rounds 5 \
    --top_k_concat 6 \
    --n_docs_retrieve 200 \
    --training_mode standard \
    --conda_env dr-env-3-9-qwen \
    --norm_query \
    --topk 100 \
    --resume
"