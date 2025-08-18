#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=h200,l40s
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=dr-gru-embed
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/retrieve.out
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python passage_retrieval.py \
    --model_name_or_path infly/inf-retriever-v1-1.5b \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/inf_retriever/qampari_base_infly/passages_*' \
    --data /scratch/dq2024/diverse_retriever/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp0.json \
    --output_dir /scratch/dq2024/diverse_retriever/training/inf_retriever/ \
    --outfile qampari_corpus_base_t200.json \
    --n_docs 200
"