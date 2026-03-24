import json
import sys


def convert_fanoutqa(input_file: str, output_jsonl: str, output_json: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    jsonl_entries = []
    json_entries = []

    skipped_no_answers = 0

    for idx, item in enumerate(data):
        question = item.get("question", "")
        qid_str = f"FanoutQA-{idx}"
        evidence = item.get("necessary_evidence", [])

        # Build answer_list from necessary_evidence titles
        answer_list = [
            {
                "answer_text": ev["title"],
                "aid": "m.00000",
                "aliases": [ev["title"]],
                "answer_url": ev.get("url", ""),
                "proof": []
            }
            for ev in evidence
            if ev.get("title")
        ]

        if not answer_list:
            skipped_no_answers += 1
            print(f"  [SKIP] Entry {idx} (id={item.get('id', '')!r}): no answers available")
            continue

        # --- File 1: JSONL ---
        jsonl_entry = {
            "question_text": question,
            "answer_list": answer_list,
            "qid": qid_str,
            "ground_truths": []
        }
        jsonl_entries.append(jsonl_entry)

        # --- File 2: JSON ---
        json_entry = {
            "question": question,
            "input": question,
            "id": idx,
            "answers": [""],
            "ctxs": [],
            "input_negative_ctxs": []
        }
        json_entries.append(json_entry)

    # Write JSONL (one entry per line, no extra spacing)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write JSON (pretty-printed with 4-space indent)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_entries, f, ensure_ascii=False, indent=4)

    print(f"\n=== Conversion Complete ===")
    print(f"Total input entries      : {len(data)}")
    print(f"Entries written          : {len(jsonl_entries)}")
    print(f"Skipped (no answers)     : {skipped_no_answers}")
    print(f"JSONL output             : {output_jsonl}")
    print(f"JSON  output             : {output_json}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_fanoutqa.py <input.json> <output.jsonl> <output.json>")
        sys.exit(1)

    convert_fanoutqa(sys.argv[1], sys.argv[2], sys.argv[3])