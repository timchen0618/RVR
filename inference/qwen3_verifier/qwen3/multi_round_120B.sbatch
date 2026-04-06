#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=q120m
#SBATCH --gres=gpu:h200:2
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/120B/out_files/pipeline.out
#SBATCH --account=torch_pr_152_courant
#SBATCH --mail-type=ALL

#SBATCH --mail-user=dq2024@nyu.edu

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif_120B.ext3:ro \
            /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 
export VLLM_GDN_PREFILL_BACKEND=triton
grep -r 'gdn_prefill' /ext3/py3.12/lib/python3.12/site-packages/vllm/;


python /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/multi_round_120B.py \
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/120B/retrieved_files/round3_retrieved.json \
    --source_queries_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/120B \
    --top_n_concat 6 \
    --round_num 3 \
    --verifier Qwen/Qwen3.5-122B-A10B \
    --batch_size 16 \
    --verify_top_k 100
"
