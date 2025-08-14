import json

def read_jsonl(file_path):
    """Read a JSONL file, skipping malformed lines."""
    data = []
    error_count = 0
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                example = json.loads(line)
                data.append(example)
            except json.JSONDecodeError:
                error_count += 1
                print(f"Warning: Skipping malformed JSON at line {line_num} in {file_path}")
                continue
    if error_count > 0:
        print(f"Total malformed lines skipped: {error_count}")
    return data

qampari_data = read_jsonl("qampari_corpus_base_finetuned_hard_neg.json")
train_data = read_jsonl("../../../train_data.jsonl")

ctx_map = {}
for item in qampari_data:
    qid = item.get("qid")
    if not qid or "ctxs" not in item:
        continue
    ctx_map[qid] = item["ctxs"][:5]

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
