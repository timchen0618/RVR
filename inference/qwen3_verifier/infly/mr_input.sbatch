#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=multi-round-pipeline
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/multi_inp/pipeline_%A.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

NUMBERS=(0 1 2 3 9 12) 

for NUM in "${NUMBERS[@]}"; do

    
    singularity exec ${nv} \
                --overlay /scratch/$USER/qwen3_verif.ext3:ro \
                /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                /bin/bash -c "
    source /ext3/env.sh; 
    conda activate /ext3/py3.12; 
    python /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/multi_round.py \
        --initial_retrieval_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp9/infly/retrieved_files/round1_retrieved.json \
        --source_queries_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
        --output_base_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/inp${NUM}/infly \
        --top_n_concat ${NUM} \
        --round_num 1 \
        --verifier Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --batch_size 16 \
        --verify_top_k 100
    "
    
    echo "Completed inp${NUM}"
done