#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --array=0,1,2,3,6,9,12,15,18,21,24,27,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100
#SBATCH --job-name=retr-AR-%a
#SBATCH --output=out_files/0-%a.out
#SBATCH --gres=gpu:1

INPUTS=$SLURM_ARRAY_TASK_ID

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

INPUT_FILE="qampari_base_finetined_infly_0-${INPUTS}_concat_200.json"
OUTPUT_FILE="qampari_corpus_base_r1inf_r2std0-3_AR_0-${INPUTS}_t200.json"

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9-qwen; 

python /scratch/dq2024/diverse_retriever/training/inf_retriever/passage_retrieval.py \
    --model_name_or_path /scratch/dq2024/models/diverse_retriever/checkpoints/qampari_infly_standard_finetuned_0-3/checkpoint/best_model \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/inference/infly/passage_embeddings/qampari_standard_finetuned_infly_0-3/passages_*' \
    --data /scratch/dq2024/diverse_retriever/inference/infly/base/input_files/$INPUT_FILE \
    --output_dir /scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model \
    --outfile $OUTPUT_FILE \
    --training_mode "standard" \
    --n_docs 100 \
    --norm_query
"