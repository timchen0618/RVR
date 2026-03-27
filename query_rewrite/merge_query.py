#!/usr/bin/env python3
"""
Merge Keywords Script

Takes a JSONL file with extracted keywords and concatenates them after the original query.

Input format (JSONL):
{"question": "...", "keywords": "term1, term2, term3", ...}

Output format (JSON array):
[{"question": "original question? term1, term2, term3", "input": "...", ...}]
"""

import json
import argparse
import os
import sys


def process_file(input_file: str, output_file: str):
    """Read JSONL file, merge query + keywords, output as JSON array"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    print(f"Loaded {len(entries)} entries")

    processed = []
    stats = {'with_keywords': 0, 'without_keywords': 0}

    for idx, entry in enumerate(entries):
        query = entry.get('question', '').strip()
        keywords = entry.get('keywords', '').strip()

        if keywords:
            if not query.endswith(('?', '.', '!')):
                query += '?'
            merged_query = f"{query} {keywords}"
            stats['with_keywords'] += 1
        else:
            merged_query = query
            stats['without_keywords'] += 1

        new_entry = {
            "question": merged_query,
            "input": merged_query,
            "id": entry.get('id', idx),
            "answers": entry.get('answers', []),
            "ctxs": [],
            "input_negative_ctxs": []
        }

        processed.append(new_entry)

    # Save as JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)

    print(f"\n=== Stats ===")
    print(f"Total entries: {len(processed)}")
    print(f"With keywords (merged): {stats['with_keywords']}")
    print(f"Without keywords (kept original): {stats['without_keywords']}")
    print(f"\nSaved to: {output_file}")

    if processed:
        print(f"\n=== Sample Output ===")
        print(json.dumps(processed[0], indent=2, ensure_ascii=False)[:300])


def main():
    parser = argparse.ArgumentParser(
        description="Merge original query with extracted keywords"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with keywords")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file with merged queries")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    process_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()