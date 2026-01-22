#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --job-name=launch-sweep
#SBATCH --output=/scratch/dq2024/diverse_retriever/training/contriever/standard/out_files/launcher.out
#SBATCH --account=torch_pr_152_courant

LRS=(0.00005 0.0001 0.0002)
TEMPS=(0.05 0.07 0.1)
WARMUP=1000
TOTAL_STEPS=10000

WORKER_SCRIPT="/scratch/dq2024/diverse_retriever/training/contriever/standard/worker_job.sh"

cat > ${WORKER_SCRIPT} << 'WORKER_EOF'
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --account=torch_pr_152_courant

LR=$1
TEMP=$2
WARMUP=$3
TOTAL_STEPS=$4

JOB_ID="lr${LR}_temp${TEMP}_warmup${WARMUP}"

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

accelerate launch --main_process_port 29502 ../finetuning.py \
    --train_data ../../../train_data_doubled.jsonl \
    --eval_data ../../../train_eval_data.jsonl \
    --temperature ${TEMP} \
    --total_steps ${TOTAL_STEPS} \
    --warmup_steps ${WARMUP} \
    --lr ${LR} \
    --save_freq 2500 \
    --log_freq 50 \
    --eval_freq 2500 \
    --negative_hard_ratio 0.0 \
    --negative_ctxs 1 \
    --per_gpu_batch_size 48 \
    --per_gpu_eval_batch_size 48 \
    --model_path facebook/contriever-msmarco \
    --chunk_length 512 \
    --accumulation_steps 2 \
    --run_name \"qampari_sweep_${JOB_ID}_steps${TOTAL_STEPS}\" \
    --output_dir \"/scratch/dq2024/models/diverse_retriever/checkpoints/sweep_${JOB_ID}/\" \
    --training_mode \"standard\" \
    --max_positive_documents 1 \
    --sample_length \
    --doc_length 1 2 3
"
WORKER_EOF

chmod +x ${WORKER_SCRIPT}

for lr in "${LRS[@]}"; do
    for temp in "${TEMPS[@]}"; do
        sbatch --job-name=sweep-lr${lr}-t${temp} \
               --output=/scratch/dq2024/diverse_retriever/training/contriever/standard/out_files/sweep-lr${lr}_temp${temp}.out \
               ${WORKER_SCRIPT} $lr $temp $WARMUP $TOTAL_STEPS
        sleep 0.5
    done
done