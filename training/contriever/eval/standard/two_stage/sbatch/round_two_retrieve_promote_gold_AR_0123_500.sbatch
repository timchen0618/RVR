#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=ret0123
#SBATCH --output=out_files/500ret0123.out
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
    --data /scratch/dq2024/diverse_retriever/training/contriever/eval/qwen3_verifier/qampari_corpus_base_only_random_inp0_t500_promote-qwen3_0-12_concat.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/eval/qwen3_verifier \
    --outfile qampari_std_t500_promote-qwen3_0-_retrieved.json \
    --training_mode "standard" \
    --n_docs 100
"
