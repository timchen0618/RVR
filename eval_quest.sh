retriever="contriever"
for _lambda in 0.5 0.6 0.7 0.8 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99
do
    echo "Running evaluation with lambda=${_lambda} and retriever=${retriever}  retrieval_outputs/quest/reranked/${retriever}_reranked_l${_lambda}.jsonl"
    python src/eval/eval_quest_ret.py \
        --pred retrieval_outputs/quest/reranked/${retriever}_reranked_l${_lambda}.jsonl \
        --gold data/quest_data/dev_data.jsonl
done