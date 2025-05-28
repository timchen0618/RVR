#!/bin/bash

# # QAMPARI Corpus
# # ORG
# python passage_retrieval.py \
#     --model_name_or_path facebook/contriever-msmarco \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_embeddings/*" \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_org.json \
#     --n_docs 100

# # Only Random 
# python passage_retrieval.py \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpusqampari_only_random_steps5000_warmup250_lr0.0001_nhr0.0_nctxs1_bz64/checkpoint/step-3500/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random/*" \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_only_random.json \
#     --n_docs 100


# # random to input
# python passage_retrieval.py \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_input_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz32/checkpoint/step-4000/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_input/*" \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_input.json \
#     --n_docs 100

# # random to cont
# python passage_retrieval.py \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --n_docs 100 \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_cont/*" \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_cont_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-10000/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_cont.json 

# # random to NV
# python passage_retrieval.py \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --n_docs 100 \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_nv/*" \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_nv_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-10000/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_nv.json 

# # Trained w/ Random Negatives -> Hard Negatives (mined by contriever) + Input Negatives
# python passage_retrieval.py \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --n_docs 100 \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_cont+input/*" \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_cont+input_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-10000/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_cont+input.json

# # Trained w/ Random Negatives -> Hard Negatives (mined by NV) + Input Negatives
# python passage_retrieval.py \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --n_docs 100 \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_nv+input/*" \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_nv+input_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-10000/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_nv+input.json 


# # Trained w/ Random Negatives + Input Negatives
# python passage_retrieval.py \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --n_docs 100 \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_random+input/*" \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpusqampari_random+input_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-8000/ \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random+input.json 


# # Subtraction (Random + Input)
# python passage_retrieval.py \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --n_docs 100 \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_subtraction_only_random+input/*" \
#     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_subtraction_only_random+input_steps10000_warmup500_lr0.0001_nhr0.25_nctxs1_bz32/checkpoint/step-9000 \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_subtraction_random+input.json \
#     --training_mode subtraction 


# # Subtraction ORG
# python passage_retrieval.py \
#     --model_name_or_path facebook/contriever-msmarco \
#     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     --passages_embeddings "/datastor1/hungting/Contriever/qampari_embeddings/*" \
#     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query.json  \
#     --output_dir outputs/contriever_msmarco_qampari/ \
#     --n_docs 100 \
#     --training_mode subtraction \
#     --outfile qampari_dev_2nd_stage_qampari_corpus_org_subtraction.json

#     # --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_subtraction_random+input.json \



###################################################
##### Reproduce Differing Number of Documents #####
###################################################
# Only Random 

# for i in 3 2
# do
#     # Standard (Only Random), Old
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random/*" \
#         --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpusqampari_only_random_steps5000_warmup250_lr0.0001_nhr0.0_nctxs1_bz64/checkpoint/step-3500/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_standard_only_random_old_inp${i}.json \
#         --training_mode standard

#     # Standard (Only Random)
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_standard_only_random/*" \
#         --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_standard_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_standard_only_random_inp${i}.json \
#         --training_mode standard


#     # Standard (Only Random), combined
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_standard_only_random_combined/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_standard_only_random_combined_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_standard_only_random_combined_inp${i}.json \
#         --training_mode standard
# done


# for i in 3
# do
#     python passage_retrieval.py \
#         --model_name_or_path facebook/contriever-msmarco \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_embeddings/*" \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_org_concat_query_inp${i}.json \
#         --n_docs 100

#     # GRU (Only Random), combined
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_gru_only_random_combined/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_gru_only_random_combined_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_gru_only_random_combined_inp${i}.json \
#         --training_mode gru
# done

# for i in {0..3}
# do

#     # python passage_retrieval.py \
#     #     --model_name_or_path facebook/contriever-msmarco \
#     #     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     #     --passages_embeddings "/datastor1/hungting/Contriever/qampari_embeddings/*" \
#     #     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#     #     --output_dir outputs/contriever_msmarco_qampari/ \
#     #     --outfile qampari_dev_2nd_stage_qampari_corpus_org_concat_query_inp${i}.json \
#     #     --n_docs 100

#     # python passage_retrieval.py \
#     #     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpusqampari_only_random_steps5000_warmup250_lr0.0001_nhr0.0_nctxs1_bz64/checkpoint/step-3500/ \
#     #     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     #     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random/*" \
#     #     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#     #     --output_dir outputs/contriever_msmarco_qampari/ \
#     #     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_only_random_inp${i}.json \
#     #     --n_docs 100

#     # # random to cont
#     # python passage_retrieval.py \
#     #     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#     #     --output_dir outputs/contriever_msmarco_qampari/ \
#     #     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     #     --n_docs 100 \
#     #     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_cont/*" \
#     #     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_cont_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-10000/ \
#     #     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_cont_inp${i}.json 

#     # # random to NV
#     # python passage_retrieval.py \
#     #     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_concat_query_inp${i}.json  \
#     #     --output_dir outputs/contriever_msmarco_qampari/ \
#     #     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     #     --n_docs 100 \
#     #     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_only_random_to_nv/*" \
#     #     --model_name_or_path /datastor1/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_only_random_to_nv_steps10000_warmup250_lr0.0001_nhr0.75_nctxs4_bz16/checkpoint/step-10000/ \
#     #     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_random_to_nv_inp${i}.json 

#     # GRU (Only Random), Old
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_gru_only_random_old/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_gru_only_random_old_steps20000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_gru_only_random_old_inp${i}.json \
#         --training_mode gru

#     # GRU (Only Random)
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_gru_only_random/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_gru_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/step-50000/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_gru_only_random_inp${i}.json \
#         --training_mode gru


#     # # GRU (Only Random), combined
#     # python passage_retrieval.py \
#     #     --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#     #     --output_dir outputs/contriever_msmarco_qampari/ \
#     #     --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#     #     --n_docs 100 \
#     #     --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_gru_only_random_combined/*" \
#     #     --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_gru_only_random_combined_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#     #     --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_gru_only_random_combined_inp${i}.json \
#     #     --training_mode gru

#     # Linear Projection (Only Random), Old
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_linear_projection_only_random_old/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_linear_projection_only_random_old_steps20000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_linear_projection_only_random_old_inp${i}.json \
#         --training_mode linear_projection

#     # Linear Projection (Only Random)
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_linear_projection_only_random/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_linear_projection_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/step-45000/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_linear_projection_only_random_inp${i}.json \
#         --training_mode linear_projection


#     # Linear Projection (Only Random), combined
#     python passage_retrieval.py \
#         --data /scratch/cluster/hungting/projects/diverse_response/data/qampari_data/nqformat_data/qampari_dev_2nd_stage_qampari_corpus_org_query_inp${i}.json  \
#         --output_dir outputs/contriever_msmarco_qampari/ \
#         --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
#         --n_docs 100 \
#         --passages_embeddings "/datastor1/hungting/Contriever/qampari_finetuned_linear_projection_only_random_combined/*" \
#         --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_linear_projection_only_random_combined_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz48/checkpoint/best_model/ \
#         --outfile qampari_dev_2nd_stage_qampari_corpus_finetuned_linear_projection_only_random_combined_inp${i}.json \
#         --training_mode linear_projection
