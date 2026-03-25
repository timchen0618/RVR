#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/dq2024/diverse_retriever/eval/out_files/mmr.out
#SBATCH --account=torch_pr_152_courant
 
 
if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi
 
 
singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh;
conda activate dr-env-3-9;
 
python /scratch/dq2024/diverse_retriever/eval/mmr.py \
    /scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t200.json \
    /scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100_mmr_from_200.json \
    100 \
    1.0
"
 