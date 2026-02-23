#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=v100qb
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/extract_test/test.out
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 
python extract_answer.py \
  --input_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/all_golds/round1_all_golds.json \
  --output_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/contriever/extract_test \
  --top_n 3 \
  --batch_size 16
"

