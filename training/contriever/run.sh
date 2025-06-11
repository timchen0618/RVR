mode=$1
for total_steps in 50000; do  # 3
    for warmup_steps in 1000; do  # 3
        for lr in 0.0001; do  # 3
            for negative_hard_ratio in 0.0; do  # 3
                for negative_ctxs in 1; do # 1
                    for per_gpu_batch_size in 48; do
                            MAX_POS=1
                            ADD_INPUT_NEGATIVES=false
                            
                            # mode="concat"
                            # OUTPUT_DIR="/scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/"
                            # OUTPUT_DIR="/datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/"
                            # OUTPUT_DIR="/scratch/hc3337/models/diverse_response/checkpoint/qampari_corpus/"
                            OUTPUT_DIR="/scratch/dq2024/models/diverse_retriever/checkpoints/"
                            PORT=29500

                            if [ "$ADD_INPUT_NEGATIVES" = true ]; then
                                echo "Adding input negatives"
                                add_input_negatives_str="--add_input_negatives"
                            else
                                echo "NOT adding input negatives"
                                add_input_negatives_str=""
                            fi

                            # accelerate launch --main_process_port $PORT finetuning.py --train_data ../data/qampari_data/qampari_corpus/train_random+input_data.jsonl  \
                            #                      --eval_data ../data/qampari_data/qampari_corpus/train_eval_random+input_data.jsonl  \
                            #                      --temperature 0.05 \
                            #                      --total_steps ${total_steps} \
                            #                      --warmup_steps ${warmup_steps} \
                            #                      --lr ${lr} \
                            #                      --save_freq 2500 \
                            #                      --log_freq 250 \
                            #                      --eval_freq 2500 \
                            #                      --negative_hard_ratio ${negative_hard_ratio} \
                            #                      --negative_ctxs ${negative_ctxs} \
                            #                      ${add_input_negatives_str} \
                            #                      --per_gpu_batch_size ${per_gpu_batch_size} \
                            #                      --per_gpu_eval_batch_size ${per_gpu_batch_size} \
                            #                      --model_path facebook/contriever-msmarco \
                            #                      --chunk_length 512 \
                            #                      --accumulation_steps 2 \
                            #                      --run_name "qampari_${mode}_only_random_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}_pos${MAX_POS}" \
                            #                      --output_dir ${OUTPUT_DIR} \
                            #                      --training_mode "${mode}" \
                            #                      --max_positive_documents ${MAX_POS} 
                            accelerate launch --main_process_port $PORT finetuning.py --train_data ../../train_random+input_data.jsonl  \
                                                 --eval_data ../../train_random+input_data.jsonl  \
                                                 --temperature 0.05 \
                                                 --total_steps ${total_steps} \
                                                 --warmup_steps ${warmup_steps} \
                                                 --lr ${lr} \
                                                 --save_freq 2500 \
                                                 --log_freq 250 \
                                                 --eval_freq 2500 \
                                                 --negative_hard_ratio ${negative_hard_ratio} \
                                                 --negative_ctxs ${negative_ctxs} \
                                                 ${add_input_negatives_str} \
                                                 --per_gpu_batch_size ${per_gpu_batch_size} \
                                                 --per_gpu_eval_batch_size ${per_gpu_batch_size} \
                                                 --model_path facebook/contriever-msmarco \
                                                 --chunk_length 512 \
                                                 --accumulation_steps 2 \
                                                 --run_name "qampari_${mode}_only_random_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}_pos${MAX_POS}" \
                                                 --output_dir ${OUTPUT_DIR} \
                                                 --training_mode "${mode}" \
                                                 --max_positive_documents ${MAX_POS} 
                    done
                done
            done
        done
    done
done



# NUM_NODE=4
# PORT=29500
# torchrun --nproc_per_node ${NUM_NODE} --master-port ${PORT}


# for total_steps in 20000; do  # 4
#     for warmup_steps in 500; do  # 3
#         for lr in 0.00001 0.000001 0.0001 0.001; do  # 4
#             for negative_hard_ratio in 0.5 0.25 0.75; do  # 3
#                 for negative_ctxs in 2 4 8; do # 2
#                     for per_gpu_batch_size in 16; do
#                             python finetuning.py --train_data ../data/qampari_data/train_data.jsonl  \
#                                                  --eval_data ../data/qampari_data/train_eval_data.jsonl  \
#                                                  --temperature 0.05 \
#                                                  --total_steps ${total_steps} \
#                                                  --warmup_steps ${warmup_steps} \
#                                                  --lr ${lr} \
#                                                  --save_freq 500 \
#                                                  --log_freq 100 \
#                                                  --eval_freq 250 \
#                                                  --negative_hard_ratio ${negative_hard_ratio} \
#                                                  --negative_ctxs ${negative_ctxs} \
#                                                  --per_gpu_batch_size ${per_gpu_batch_size} \
#                                                  --per_gpu_eval_batch_size 256 \
#                                                  --model_path facebook/contriever-msmarco \
#                                                  --chunk_length 512 \
#                                                  --output_dir /datastor1/hungting/models/diverse_response/checkpoint 
#                     done
#                 done
#             done
#         done
#     done
# done

# for total_steps in 20000; do  # 4
#     for warmup_steps in 500; do  # 3
#         for lr in 0.00001; do  # 4
#             for negative_hard_ratio in 0.5; do  # 3
#                 for negative_ctxs in 2; do # 2
#                     for per_gpu_batch_size in 16; do
#                             python finetuning.py --train_data ../data/qampari_data/train_data.jsonl  \
#                                                  --eval_data ../data/qampari_data/train_eval_data.jsonl  \
#                                                  --temperature 0.05 \
#                                                  --total_steps ${total_steps} \
#                                                  --warmup_steps ${warmup_steps} \
#                                                  --lr ${lr} \
#                                                  --save_freq 500 \
#                                                  --log_freq 100 \
#                                                  --eval_freq 250 \
#                                                  --negative_hard_ratio ${negative_hard_ratio} \
#                                                  --negative_ctxs ${negative_ctxs} \
#                                                  --per_gpu_batch_size ${per_gpu_batch_size} \
#                                                  --per_gpu_eval_batch_size 256 \
#                                                  --model_path facebook/contriever-msmarco \
#                                                  --chunk_length 512 \
#                                                  --run_name "subtraction_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}" \
#                                                  --output_dir /datastor1/hungting/models/diverse_response/checkpoint \
#                                                  --training_mode "subtraction"
#                     done
#                 done
#             done
#         done
#     done
# done








# for total_steps in 100000; do  # 3
#     for warmup_steps in 1000; do  # 3
#         for lr in 0.0001; do  # 3
#             for negative_hard_ratio in 0.0; do  # 3
#                 for negative_ctxs in 1; do # 1
#                     for per_gpu_batch_size in 8; do
#                             ADD_INPUT_NEGATIVES=false
#                             mode="linear_projection"
#                             # mode="concat"
#                             OUTPUT_DIR="/scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/"
#                             # OUTPUT_DIR="/datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/"
#                             # OUTPUT_DIR="/scratch/hc3337/models/diverse_response/checkpoint/qampari_corpus/"
#                             PORT=29500

#                             if [ "$ADD_INPUT_NEGATIVES" = true ]; then
#                                 echo "Adding input negatives"
#                                 add_input_negatives_str="--add_input_negatives"
#                             else
#                                 echo "NOT adding input negatives"
#                                 add_input_negatives_str=""
#                             fi

#                             accelerate launch --main_process_port $PORT finetuning.py --train_data ../data/qampari_data/qampari_corpus/train_data.jsonl  \
#                                                  --eval_data ../data/qampari_data/qampari_corpus/train_eval_data.jsonl  \
#                                                  --temperature 0.05 \
#                                                  --total_steps ${total_steps} \
#                                                  --warmup_steps ${warmup_steps} \
#                                                  --lr ${lr} \
#                                                  --save_freq 2500 \
#                                                  --log_freq 250 \
#                                                  --eval_freq 2500 \
#                                                  --negative_hard_ratio ${negative_hard_ratio} \
#                                                  --negative_ctxs ${negative_ctxs} \
#                                                  ${add_input_negatives_str} \
#                                                  --per_gpu_batch_size ${per_gpu_batch_size} \
#                                                  --per_gpu_eval_batch_size ${per_gpu_batch_size} \
#                                                  --model_path facebook/contriever-msmarco \
#                                                  --chunk_length 512 \
#                                                  --accumulation_steps 8 \
#                                                  --run_name "qampari_${mode}_only_random_sampled_length_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}" \
#                                                  --output_dir ${OUTPUT_DIR} \
#                                                  --training_mode "${mode}" \
#                                                  --sample_length
#                     done
#                 done
#             done
#         done
#     done
# done



# id=$1
# free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

# while [ $free_mem -lt 40000 ]
# do
#     free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
#     sleep 10
# done

# for total_steps in 20000; do  # 3
#     for warmup_steps in 1000; do  # 3
#         for lr in 0.0001; do  # 3
#             for negative_hard_ratio in 0.0; do  # 3
#                 for negative_ctxs in 1; do # 1
#                     for per_gpu_batch_size in 64; do
#                             ADD_INPUT_NEGATIVES=false
#                             mode="standard_org_q"
#                             # mode="concat"
#                             OUTPUT_DIR="/scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/"
#                             # OUTPUT_DIR="/datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/"
#                             # OUTPUT_DIR="/scratch/hc3337/models/diverse_response/checkpoint/qampari_corpus/"
#                             PORT=29400

#                             if [ "$ADD_INPUT_NEGATIVES" = true ]; then
#                                 echo "Adding input negatives"
#                                 add_input_negatives_str="--add_input_negatives"
#                             else
#                                 echo "NOT adding input negatives"
#                                 add_input_negatives_str=""
#                             fi

#                             accelerate launch --main_process_port $PORT finetuning.py --train_data ../data/qampari_data/qampari_corpus/train_data.jsonl  \
#                                                  --eval_data ../data/qampari_data/qampari_corpus/train_eval_data.jsonl  \
#                                                  --temperature 0.05 \
#                                                  --total_steps ${total_steps} \
#                                                  --warmup_steps ${warmup_steps} \
#                                                  --lr ${lr} \
#                                                  --save_freq 1250 \
#                                                  --log_freq 250 \
#                                                  --eval_freq 1250 \
#                                                  --negative_hard_ratio ${negative_hard_ratio} \
#                                                  --negative_ctxs ${negative_ctxs} \
#                                                  ${add_input_negatives_str} \
#                                                  --per_gpu_batch_size ${per_gpu_batch_size} \
#                                                  --per_gpu_eval_batch_size ${per_gpu_batch_size} \
#                                                  --model_path facebook/contriever-msmarco \
#                                                  --chunk_length 512 \
#                                                  --accumulation_steps 2 \
#                                                  --run_name "qampari_${mode}_only_random_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}" \
#                                                  --output_dir ${OUTPUT_DIR} \
#                                                  --training_mode "${mode}" 
#                     done
#                 done
#             done
#         done
#     done
# done