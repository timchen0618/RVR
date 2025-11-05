python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/contriever/gold_documents/round1_all_gold_docs.json \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/contriever/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv

python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/contriever/base/gold_documents/round1_all_gold_docs.json \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv


python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max3/gold_documents/round1_all_gold_docs.json \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max3/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv

python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max3/base/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv

python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max6/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv

python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max6/base/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv

python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max12/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv

python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/qwen3/max12/base/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv