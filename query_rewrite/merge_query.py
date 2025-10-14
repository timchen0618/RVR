import json


# File paths - modify these as needed
INPUT_FILE = "/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qampari_qwen3_keywords.json"
OUTPUT_FILE = "/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qampari_qwen3_original+keywords.json"


def merge_queries(input_file, output_file):
    """
    Read JSONL file and merge original_query with question field.
    """
    processed_data = []
    
    # Read JSONL (one JSON object per line)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                obj = json.loads(line)
                merge_single_object(obj)
                processed_data.append(obj)
    
    # Write as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in processed_data:
            json.dump(obj, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully merged {len(processed_data)} objects and saved to {output_file}")


def merge_single_object(obj):
    """
    Merge original_query and question fields in a single object.
    """
    if 'original_query' in obj and 'question' in obj:
        # Merge the two fields with a space between them
        original = obj['question'].strip()
        question = obj['keywords'].strip()
        
        # Ensure proper spacing and punctuation
        if not original.endswith(('?', '.', '!')):
            original += '?'
        
        # Merge the fields
        obj['question'] = f"{original} {question}"
    
    return obj


def main():
    merge_queries(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()