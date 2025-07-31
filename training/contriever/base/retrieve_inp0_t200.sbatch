#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=h200
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=dr02-base-retrieve
#SBATCH --output=base-retrieve-inp0_200.out
#SBATCH --gres=gpu:1
#SBATCH --wrap "hostname && sleep infinity"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/contriever/qampari_base_only_random/passages_*' \
    --data /scratch/dq2024/diverse_retriever/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp0.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/contriever/contriever_msmarco_qampari \
    --outfile qampari_corpus_base_only_random_inp0_t200.json \
    --n_docs 200
"