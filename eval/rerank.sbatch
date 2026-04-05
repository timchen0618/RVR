#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=eval
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/out_files/rerank_inf.out
#SBATCH --account=torch_pr_152_courant
#SBATCH --gres=gpu:1


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/qwen3_verif.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate /ext3/py3.12; 

python eval/rerank.py \
    --input_file /scratch/dq2024/diverse_retriever/eval/infly/base/qampari_corpus_finetuned_base_t500.json \
    --output_file /scratch/dq2024/diverse_retriever/eval/infly/base/qampari_corpus_finetuned_base_t400_reranked.json \
    --top_k 100 \
    --batch_size 16
"
