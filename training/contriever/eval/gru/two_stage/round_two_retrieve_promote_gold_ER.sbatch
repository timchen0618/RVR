#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=h200
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=dr02-base-retrieve
#SBATCH --output=out_files/gru-retrieve-ER.out
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
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_gru_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48_pos1_2gpu/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/contriever/qampari_finetuned_gru_only_random/passages_*' \
    --data /scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/promote_gold/qampari_corpus_finetuned_gru_only_random_promote_gold_ER.json \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage \
    --outfile qampari_corpus_base_only_random_promote_gold_ER_round_two.json \
    --training_mode "gru" \
    --n_docs 100
"