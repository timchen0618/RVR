#!/usr/bin/env python3
"""
Script to replace question fields in one JSONL file with questions from another JSONL file.
Matches entries by index (line number).
"""

import json

# File paths - EDIT THESE
SOURCE_FILE = "/scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t200.json"  # Questions you WANT
TARGET_FILE = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_std_t200_promote-qwen3_0-3_retrieved_200.json"              # File to UPDATE
OUTPUT_FILE = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/test.json"                 # Where to save result


def main():
    print("Loading source questions...")
    source_questions = []
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    question = entry.get('question', entry.get('question_text', ''))
                    source_questions.append(question)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num} in source file")
                    source_questions.append(None)  # Keep index alignment
    
    print(f"Loaded {len(source_questions)} questions from source file")
    
    print("\nUpdating target file...")
    updated_entries = []
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    
                    # Get the corresponding question from source file (0-indexed)
                    idx = line_num - 1
                    if idx < len(source_questions) and source_questions[idx] is not None:
                        # Replace the question field
                        entry['question'] = source_questions[idx]
                        
                        # Also update 'input' field if it exists
                        if 'input' in entry:
                            entry['input'] = source_questions[idx]
                        
                        # Also update 'question_text' field if it exists
                        if 'question_text' in entry:
                            entry['question_text'] = source_questions[idx]
                    else:
                        print(f"Warning: No matching question for line {line_num}")
                    
                    updated_entries.append(entry)
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num} in target file")
                    continue
    
    print(f"Updated {len(updated_entries)} entries")
    
    print("\nWriting output file...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nDone! Output written to: {OUTPUT_FILE}")
    print(f"Total entries: {len(updated_entries)}")


if __name__ == "__main__":
    main()