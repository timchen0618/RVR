import os
import glob
import json
import csv

# adjust this if your folder lives somewhere else
INPUT_DIR = "wikipedia_chunks/chunks_v5"
OUTPUT_PATH = "./chunks_v5.tsv"

# make sure output dir exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out_tsv:
    writer = csv.writer(out_tsv, delimiter="\t")
    # optionally write a header:
    # writer.writerow(["id", "text", "title"])

    # glob and sort so chunks go in order
    pattern = os.path.join(INPUT_DIR, "wikipedia_chunks_*.jsonl")
    for fn in sorted(glob.glob(pattern)):
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["id"]
                title  = data["meta"]["title"]
                text   = data["contents"].replace("\n", " ")  # flatten newlines
                writer.writerow([doc_id, text, title])

print(f"Wrote TSV to {OUTPUT_PATH}")
