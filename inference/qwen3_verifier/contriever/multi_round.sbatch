#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=multi-round-pipeline
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/contriever/out_files/pipeline.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 
python /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/multi_round.py \
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/contriever/retrieved_files/round2_retrieved.json \
    --source_queries_file /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t100.json \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v50/contriever \
    --top_n_concat 3 \
    --round_num 2 \
    --verifier Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --batch_size 16 \
    --verify_top_k 20 \
    --unique_input
"

