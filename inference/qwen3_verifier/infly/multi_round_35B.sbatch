#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=300GB
#SBATCH --job-name=i35
#SBATCH --gres=gpu:h200:1
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/35B/out_files/pipeline.out
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
python /scratch/dq2024/diverse_retriever/inference/qwen3_verifier/multi_round_120B.py \
    --initial_retrieval_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/retrieved_files/round1_retrieved.json \
    --source_queries_file /scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --output_base_dir /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/infly/35B \
    --top_n_concat 6 \
    --round_num 1 \
    --verifier Qwen/Qwen3.5-35B-A3B \
    --batch_size 16 \
    --verify_top_k 100
"

