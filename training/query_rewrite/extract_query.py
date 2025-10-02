import json

# Configuration - Set your file names here
INPUT_JSON_FILE = "qampari_qwen3_rewritten_queries_and_keywords_llm_out.json"  # Replace with your JSON file name
OUTPUT_COLUMN1_FILE = "original.txt"  # Output for original_query
OUTPUT_COLUMN2_FILE = "rewritten.txt"  # Output for question

def extract_fields_to_columns(json_file, col1_file, col2_file):
    """
    Extract 'original_query' and 'question' fields from JSON objects
    and save them to separate text files for Excel columns.
    """
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            # Try to load as a JSON array first
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]  # Convert single object to list
            except json.JSONDecodeError:
                # If that fails, try reading line by line (JSONL format)
                f.seek(0)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            data.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON line: {e}")
                            continue
        
        if not data:
            print("Error: No valid JSON data found in the file.")
            return
        
        # Extract fields
        original_queries = []
        questions = []
        
        for i, obj in enumerate(data):
            # Extract original_query
            original_query = obj.get('original_query', '')
            # Remove newlines and extra spaces for Excel compatibility
            original_query = ' '.join(original_query.split())
            original_queries.append(original_query)
            
            # Extract question
            question = obj.get('question', '')
            # Remove newlines and extra spaces for Excel compatibility
            question = ' '.join(question.split())
            questions.append(question)
        
        # Write to column 1 file (original_query)
        with open(col1_file, 'w', encoding='utf-8') as f:
            for query in original_queries:
                f.write(query + '\n')
        
        # Write to column 2 file (question)
        with open(col2_file, 'w', encoding='utf-8') as f:
            for question in questions:
                f.write(question + '\n')
        
        # Print summary
        print(f"✓ Successfully processed {len(data)} JSON objects")
        print(f"✓ Column 1 (original_query) saved to: {col1_file}")
        print(f"✓ Column 2 (question) saved to: {col2_file}")
        print(f"\nYou can now:")
        print(f"1. Open {col1_file} and copy all content (Ctrl+A, Ctrl+C)")
        print(f"2. Paste into Excel column A")
        print(f"3. Open {col2_file} and copy all content (Ctrl+A, Ctrl+C)")
        print(f"4. Paste into Excel column B")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found. Please check the file name and path.")
    except Exception as e:
        print(f"Error processing file: {e}")

# Run the extraction
if __name__ == "__main__":
    extract_fields_to_columns(INPUT_JSON_FILE, OUTPUT_COLUMN1_FILE, OUTPUT_COLUMN2_FILE)