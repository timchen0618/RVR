#!/usr/bin/env python3
"""
Script to reformat JSONL data from the second format to the first format.
Converts question_text field to question and input fields.
"""

import json
import argparse


def reformat_entry(entry, entry_id):
    """
    Convert a single entry from the source format to the target format.
    
    Args:
        entry: Dictionary with 'question_text' field
        entry_id: Integer ID for the entry
    
    Returns:
        Dictionary in the target format
    """
    question_text = entry.get("question_text", "")
    
    reformatted = {
        "question": question_text,
        "input": question_text,
        "id": entry_id,
        "answers": [
            ""
        ],
        "ctxs": [],
        "input_negative_ctxs": []
    }
    
    return reformatted


def main():
    parser = argparse.ArgumentParser(
        description="Reformat JSONL data to match the target JSON format"
    )
    parser.add_argument(
        "input_file",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "output_file",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Read input JSONL file
    reformatted_data = []
    error_count = 0
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # Start counting from 1 for line numbers
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    entry = json.loads(line)
                    reformatted_entry = reformat_entry(entry, len(reformatted_data))
                    reformatted_data.append(reformatted_entry)
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Warning: Skipping malformed JSON at line {line_num} in {args.input_file}")
                    if error_count <= 5:  # Print first 5 problematic lines for debugging
                        print(f"Problematic line preview (first 200 chars): {line[:200]}...")
                    continue
    
    # Write output JSON file with proper formatting
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(reformatted_data, f, indent=4)
    
    print(f"Successfully reformatted {len(reformatted_data)} entries")
    print(f"Output written to: {args.output_file}")


if __name__ == "__main__":
    main()