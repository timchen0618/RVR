#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=dr-qbase-retrieve
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=base-retrieve.out
#SBATCH --gres=gpu:1

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

python ../passage_retrieval.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --passages /scratch/dq2024/diverse_retriever/chunks_v5.tsv \
    --passages_embeddings '/scratch/dq2024/diverse_retriever/training/qwen2/qampari_base/passages_*' \
    --data /scratch/dq2024/diverse_retriever/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp0.json  \
    --output_dir /scratch/dq2024/diverse_retriever/training/qwen2/base \
    --outfile qampari_corpus_base_t200.json \
    --n_docs 200
"