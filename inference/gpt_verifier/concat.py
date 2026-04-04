#!/usr/bin/env python3
"""
Convert a JSONL file with retrieved ctxs into the concatenated format
where top-6 documents are inlined into the question/input fields.

Usage:
    python convert_to_concatenated.py <input_jsonl> <output_json> [--top_k 6]
"""

import json
import argparse


def convert(input_path, output_path, top_k=6):
    results = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            question = item["question"]
            ctxs = item.get("ctxs", [])[:top_k]

            # Concatenate documents: "Title Text\nTitle Text\n..."
            doc_texts = []
            for ctx in ctxs:
                doc_texts.append(f"{ctx['title']} {ctx['text']}")
            concatenated_docs = "\n".join(doc_texts)

            # Build the combined question+documents string
            combined = f"Question: {question}\n\nDocuments: {concatenated_docs}"

            results.append({
                "question": combined,
                "input": combined,
                "id": item["id"],
                "answers": item.get("answers", [""]),
                "ctxs": [],
                "input_negative_ctxs": []
            })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Converted {len(results)} items, saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Input JSONL file with ctxs")
    parser.add_argument("output_json", help="Output JSON file (concatenated format)")
    parser.add_argument("--top_k", type=int, default=6, help="Number of top ctxs to concatenate (default: 6)")
    args = parser.parse_args()
    convert(args.input_jsonl, args.output_json, args.top_k)