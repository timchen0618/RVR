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
                            OUTPUT_DIR="/scratch/hc3337/models/diverse_response/checkpoint/qampari_corpus/"
                            PORT=29500

                            if [ "$ADD_INPUT_NEGATIVES" = true ]; then
                                echo "Adding input negatives"
                                add_input_negatives_str="--add_input_negatives"
                            else
                                echo "NOT adding input negatives"
                                add_input_negatives_str=""
                            fi

                            accelerate launch --main_process_port $PORT finetuning.py --train_data ../data/qampari_data/qampari_corpus/train_data.jsonl  \
                                                 --eval_data ../data/qampari_data/qampari_corpus/train_eval_data.jsonl  \
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
                                                 --run_name "qampari_${mode}_sample_length_036_only_random_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}_pos${MAX_POS}" \
                                                 --output_dir ${OUTPUT_DIR} \
                                                 --training_mode "${mode}" \
                                                 --max_positive_documents ${MAX_POS} \
                                                 --sample_length \
                                                 --doc_lengths 0 3 6
                    done
                done
            done
        done
    done
done