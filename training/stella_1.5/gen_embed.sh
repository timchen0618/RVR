# Subtraction (only random)
for i in {0..15}
do
    python generate_passage_embeddings.py \
        --model_name_or_path /scratch/cluster/hungting/models/diverse_response/checkpoint/qampari_corpus/qampari_subtraction_only_random_steps50000_warmup1000_lr0.0001_nhr0.0_nctxs1_bz60/checkpoint/step-15000 \
        --output_dir /datastor1/hungting/Contriever/qampari_finetuned_subtraction_only_random/  \
        --passages /datastor1/hungting/wikipedia_chunks/chunks_v5.tsv \
        --shard_id $i --num_shards 16 
done