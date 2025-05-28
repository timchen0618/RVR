#!/bin/bash
for model_path in "meta-llama/Meta-Llama-3.1-8B-Instruct"; do
# for model_path in "meta-llama/Meta-Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "gpt-4o-mini"; do
    if [ "$model_path" == "google/gemma-2-9b-it" ]; then
        system_cmd=""
    else
        system_cmd="--support_system_prompt"
    fi

    # if [ "$model_path" == "gpt-4o-mini" ]; then
    #     openai_str="_openai"
    # else
    #     openai_str=""
    # fi
    openai_str="_openai"

    # for dataset in "debateqa" "vk" "timchen0618/Arguana" "timchen0618/Kialo" "timchen0618/OpinionQA"; do
    # for dataset in "quest"; do
    #     python3 inferece.py \
    #         --model_path $model_path \
    #         --dataset $dataset \
    #         --split "test" \
    #         --temperature 0.0 \
    #         --top_p 0.9 \
    #         --instruction_file "instruction_$dataset$openai_str.txt" \
    #         $system_cmd \
    #         --max_tokens 1024 
    # done
    for dataset in "qampari"; do
        python3 inferece.py \
                --model_path $model_path \
                --dataset $dataset \
                --split "test" \
                --temperature 0.0 \
                --top_p 0.9 \
                --instruction_file "instruction_$dataset$openai_str.txt" \
                $system_cmd \
                --max_tokens 1024 \
                --use_outline
    done
    # for dataset in "qampari"; do
    #     for ret_doc_type in "bm25" 
    #     do
    #         python3 inferece.py \
    #             --model_path $model_path \
    #             --dataset $dataset \
    #             --split "test" \
    #             --temperature 0.0 \
    #             --top_p 0.9 \
    #             --instruction_file "instruction_$dataset$openai_str.txt" \
    #             $system_cmd \
    #             --max_tokens 1024 \
    #             --use_rag \
    #             --retrieved_doc_type $ret_doc_type \
    #             --use_outline
    #     done
    # done
    
done

# meta-llama/Meta-Llama-3.1-8B-Instruct
# google/gemma-2-9b-it
# debateqa, vk, timchen0618/Arguana, timchen0618/Kialo, timchen0618/OpinionQA