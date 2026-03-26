#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=16GB
#SBATCH --job-name=negate
#SBATCH --output=sbatch/out_files/negate.out
#SBATCH --account=torch_pr_152_courant

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12;

python query_rewrite/negate_query.py \
  --input_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/verified_input/keywords.json \
  --output_file /scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/qwen3_verifier/v100/qwen3/verified_input/negated_prompts.json
"