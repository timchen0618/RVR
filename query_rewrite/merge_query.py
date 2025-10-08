import json


# File paths - modify these as needed
INPUT_FILE = "/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qampari_qwen3_negated_queries.json"
OUTPUT_FILE = "/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qampari_qwen3_original+negated_queries.json"


def merge_queries(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process the data
    if isinstance(data, list):
        # If it's a list of objects
        for item in data:
            merge_single_object(item)
    elif isinstance(data, dict):
        # If it's a single object
        merge_single_object(data)
    else:
        raise ValueError("JSON must contain either a dict or a list of dicts")
    
    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully merged queries and saved to {output_file}")


def merge_single_object(obj):
    """
    Merge original_query and question fields in a single object.
    
    Args:
        obj: Dictionary object to process
    """
    if 'original_query' in obj and 'question' in obj:
        # Merge the two fields with a space between them
        original = obj['original_query'].strip()
        question = obj['question'].strip()
        
        # Ensure proper spacing and punctuation
        if not original.endswith(('?', '.', '!')):
            original += '?'
        
        # Merge the fields
        obj['question'] = f"{original} {question}"
    
    return obj


def main():
    try:
        merge_queries(INPUT_FILE, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()