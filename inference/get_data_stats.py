import json

data = []
error_count = 0
with open('../train_data.jsonl', 'r') as file:
    for line_num, line in enumerate(file, 1):
        try:
            example = json.loads(line)
            data.append(example)
        except json.JSONDecodeError:
            error_count += 1
            print(f"Warning: Skipping malformed JSON at line {line_num}")
            continue
if error_count > 0:
    print(f"Total malformed lines skipped: {error_count}")

n_golds = [len(ex.get('ground_truths', [])) for ex in data]
n_negs = [len(ex.get('negative_ctxs', [])) for ex in data]
n_hard_negs = [len(ex.get('hard_negative_ctxs', [])) for ex in data]

print(f"Avg gold docs: {sum(n_golds)/len(n_golds):.2f}")
print(f"Avg negative docs: {sum(n_negs)/len(n_negs):.2f}")
print(f"Avg hard negative docs: {sum(n_hard_negs)/len(n_hard_negs):.2f}")