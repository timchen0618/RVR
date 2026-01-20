import json

with open('../train_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

n_golds = [len(ex.get('ground_truths', [])) for ex in data]
n_negs = [len(ex.get('negative_ctxs', [])) for ex in data]
n_hard_negs = [len(ex.get('hard_negative_ctxs', [])) for ex in data]

print(f"Avg gold docs: {sum(n_golds)/len(n_golds):.2f}")
print(f"Avg negative docs: {sum(n_negs)/len(n_negs):.2f}")
print(f"Avg hard negative docs: {sum(n_hard_negs)/len(n_hard_negs):.2f}")