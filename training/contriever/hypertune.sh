# start from random
for total_steps in 5000 10000 20000; do  # 3
    for warmup_steps in 250 500 1000; do  # 2
        for lr in 0.0001 0.00001 0.001; do  # 3
            for negative_hard_ratio in 0.25 0.5 0.75; do  # 3
                for negative_ctxs in 1; do # 1
                    for per_gpu_batch_size in 32; do
                            accelerate launch --main_process_port 29600 finetuning.py --train_data ../data/qampari_data/qampari_corpus/train_random+input_data.jsonl  \
                                                 --eval_data ../data/qampari_data/qampari_corpus/train_eval_random+input_data.jsonl  \
                                                 --temperature 0.05 \
                                                 --total_steps ${total_steps} \
                                                 --warmup_steps ${warmup_steps} \
                                                 --lr ${lr} \
                                                 --save_freq 250 \
                                                 --log_freq 50 \
                                                 --eval_freq 250 \
                                                 --negative_hard_ratio ${negative_hard_ratio} \
                                                 --negative_ctxs ${negative_ctxs} \
                                                 --per_gpu_batch_size ${per_gpu_batch_size} \
                                                 --per_gpu_eval_batch_size 256 \
                                                 --model_path facebook/contriever-msmarco \
                                                 --chunk_length 512 \
                                                 --accumulation_steps 4 \
                                                 --run_name "qampari_subtraction_only_random+input" \
                                                 --output_dir /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/ \
                                                 --training_mode "subtraction" 
                    done
                done
            done
        done
    done
done


# # start from random
# for total_steps in 10000; do  # 3
#     for warmup_steps in 250; do  # 2
#         for lr in 0.0001; do  # 4
#             for negative_hard_ratio in 0.75; do  # 3
#                 for negative_ctxs in 4; do # 4  (4 8 later)
#                     for per_gpu_batch_size in 16; do
#                             accelerate launch --main_process_port 29600 finetuning.py --train_data ../data/qampari_data/qampari_corpus/train_random+nv+input_data.jsonl  \
#                                                  --eval_data ../data/qampari_data/qampari_corpus/train_eval_random+nv+input_data.jsonl  \
#                                                  --temperature 0.05 \
#                                                  --total_steps ${total_steps} \
#                                                  --warmup_steps ${warmup_steps} \
#                                                  --lr ${lr} \
#                                                  --save_freq 250 \
#                                                  --log_freq 50 \
#                                                  --eval_freq 250 \
#                                                  --negative_hard_ratio ${negative_hard_ratio} \
#                                                  --negative_ctxs ${negative_ctxs} \
#                                                  --per_gpu_batch_size ${per_gpu_batch_size} \
#                                                  --per_gpu_eval_batch_size 256 \
#                                                  --model_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpusqampari_only_random_steps5000_warmup250_lr0.0001_nhr0.0_nctxs1_bz64/checkpoint/step-3500 \
#                                                  --chunk_length 512 \
#                                                  --accumulation_steps 8 \
#                                                  --run_name "qampari_only_random_to_nv+input_steps${total_steps}_warmup${warmup_steps}_lr${lr}_nhr${negative_hard_ratio}_nctxs${negative_ctxs}_bz${per_gpu_batch_size}" \
#                                                  --output_dir /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/ \
#                                                  --training_mode "standard"
#                     done
#                 done
#             done
#         done
#     done
# done