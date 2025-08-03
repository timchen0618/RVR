#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=h200,l40s
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=r2er-retrieve
#SBATCH --output=out_files/r2er-std-ret.out
#SBATCH --gres=gpu:1
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../../../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_standard_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48_pos1_3/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/contriever/qampari_finetuned_standard_only_random_3/passages_*' \
    --data /scratch/dq2024/diverse_retriever/training/contriever/eval/standard/two_stage/vanilla_top3/qampari_standard_only_rand_gold_ER_3.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/eval/standard \
    --outfile qampari_finetuned_standard_only_rand_gold_ER_r2_3.json \
    --training_mode "standard" \
    --n_docs 100
"