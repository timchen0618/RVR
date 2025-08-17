#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=lpvr2-ret
#SBATCH --output=out_files/lp-retrieve-v-0123.out
#SBATCH --gres=gpu:1
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 


python ../../../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/torch_backup/models/qampari_contriever_lin_proj_only_random_0123/checkpoint/best_model \
    --passages /scratch/dq2024/torch_backup/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/torch_backup/diverse_retriever/training/contriever/qampari_contriever_finetuned_lin_proj_only_random_0123/passages_*' \
    --data /scratch/dq2024/torch_backup/diverse_retriever/training/contriever/eval/linear_projection/two_stage/vanilla_top0123/qampari_lin_proj_only_rand_vanilla_top0123_25_25_25_25.json \
    --output_dir /scratch/dq2024/torch_backup/diverse_retriever/training/contriever/eval/linear_projection/two_stage \
    --outfile qampari_lin_proj_only_rand_vanilla_r2_0123_25_25_25_25.json \
    --training_mode "linear_projection" \
    --n_docs 100
"
