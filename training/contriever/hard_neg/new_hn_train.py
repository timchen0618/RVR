import json

with open("qampari_corpus_base_finetuned_hard_neg.json", "r") as f:
    qampari_data = [json.loads(line) for line in f]

with open("../../../train_data.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

ctx_map = {}
for item in qampari_data:
    qid = item.get("qid")
    top_5_ctxs = item["ctxs"][:5]
    ctx_map[qid] = top_5_ctxs

updated_train_data = []
for i, entry in enumerate(train_data, 1):
    qid = entry.get("qid")
    if qid in ctx_map:
        entry["hard_negative_ctxs"] = ctx_map[qid]
    updated_train_data.append(entry)
    if i % 1000 == 0:
        print(f"Processed {i} entries")

with open("train_data_hard_neg.jsonl", "w") as f:
    for entry in updated_train_data:
        f.write(json.dumps(entry) + "\n")

print("New file saved as train_data_hard_neg.jsonl")
