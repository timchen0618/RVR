import json

# Configuration - Set your file name here
INPUT_FILE = "qampari_contriever_rewritten_queries_expansion.json"  # Replace with your JSON file name

def calculate_average_lengths(json_file):
    """
    Calculate average lengths of 'question' and 'original_query' fields
    """
    question_lengths = []
    original_query_lengths = []
    
    try:
        # Read the JSON file (JSONL format)
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    
                    # Get lengths of question and original_query fields
                    question = obj.get('question', '')
                    original_query = obj.get('original_query', '')
                    
                    question_lengths.append(len(question))
                    original_query_lengths.append(len(original_query))
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        # Calculate averages
        if question_lengths:
            avg_question = sum(question_lengths) / len(question_lengths)
            avg_original = sum(original_query_lengths) / len(original_query_lengths)
            
            # Print results
            print(f"{avg_question:.2f}")
            print(f"{avg_original:.2f}")
        else:
            print("Error: No valid data found in the file.")
            
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")

# Run the calculation
if __name__ == "__main__":
    calculate_average_lengths(INPUT_FILE)