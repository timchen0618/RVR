#!/usr/bin/env python3
"""
Negate Keywords Script

Takes a JSONL file with extracted keywords and creates negated retrieval prompts.

Input format (JSONL):
{"question": "...", "keywords": "term1, term2, term3", ...}

Output format (JSONL):
{"question": "Retrieve documents that answer the query, but do NOT retrieve documents about term1 or term2 or term3, as these answers are already covered.\n\nQuery: ...", ...}
"""

import json
import argparse
import os
import sys


def build_negation_prompt(query: str, keywords: str) -> str:
    """Build a negation prompt from query and keyword string"""
    # Split keywords by comma and clean up
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    
    if not keyword_list:
        return query
    
    negation_list = " or ".join(keyword_list)
    
    return (
        f"Retrieve documents that answer the query, "
        f"but do NOT retrieve documents about "
        f"{negation_list}, "
        f"as these answers are already covered.\n\n"
        f"Query: {query}"
    )


def process_file(input_file: str, output_file: str):
    """Process JSONL file and create negated prompts"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    
    print(f"Loaded {len(entries)} entries")
    
    processed = []
    stats = {
        'total': len(entries),
        'with_keywords': 0,
        'without_keywords': 0
    }
    
    for idx, entry in enumerate(entries):
        query = entry.get('question', '')
        keywords = entry.get('keywords', '')
        
        if keywords and keywords.strip():
            negated_query = build_negation_prompt(query, keywords)
            stats['with_keywords'] += 1
        else:
            negated_query = query
            stats['without_keywords'] += 1
        
        new_entry = {
            "question": negated_query,
            "original_query": query,
            "keywords": keywords,
            "input": negated_query,
            "id": entry.get('id', idx),
            "answers": entry.get('answers', []),
        }
        
        if "input_negative_ctxs" in entry:
            new_entry["input_negative_ctxs"] = entry["input_negative_ctxs"]
        
        processed.append(new_entry)
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n=== Stats ===")
    print(f"Total entries: {stats['total']}")
    print(f"With keywords (negated): {stats['with_keywords']}")
    print(f"Without keywords (kept original): {stats['without_keywords']}")
    print(f"\nSaved to: {output_file}")
    
    # Show sample
    if processed:
        sample = next((e for e in processed if e['keywords']), processed[0])
        print(f"\n=== Sample Output ===")
        print(f"Original: {sample['original_query']}")
        print(f"Keywords: {sample['keywords']}")
        print(f"Negated:  {sample['question'][:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Create negated retrieval prompts from extracted keywords"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with keywords")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file with negated prompts")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    process_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()