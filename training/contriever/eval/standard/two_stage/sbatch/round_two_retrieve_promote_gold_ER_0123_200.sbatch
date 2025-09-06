#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1

#SBATCH --time=1:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=stdERr2-retrieve
#SBATCH --output=out_files/r2ar-std-ret-200.out
#SBATCH --gres=gpu:1
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../../../../passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_contriever_standard_only_random_0123/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/contriever/qampari_contriever_finetuned_standard_only_random_0123/passages_*' \
    --data /scratch/dq2024/diverse_retriever/training/contriever/eval/base/base_top0123/qampari_base_finetuned_only_rand_gold_ER_200_top0123_concat.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/eval/standard \
    --outfile qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_200.json \
    --training_mode "standard" \
    --n_docs 100
"

