import json
import random

def double_training_data(input_file, output_file):
    data = []
    error_count = 0
    with open(input_file, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                example = json.loads(line)
                data.append(example)
            except json.JSONDecodeError:
                error_count += 1
                print(f"Warning: Skipping malformed JSON at line {line_num}")
                continue
    if error_count > 0:
        print(f"Total malformed lines skipped: {error_count}")
    
    doubled_data = []
    
    for example in data:
        # Version 1: Base retrieval - force 0 docs
        base_version = example.copy()
        base_version['force_base_retrieval'] = True  # Boolean flag
        doubled_data.append(base_version)
        
        # Version 2: Contextualized retrieval - use normal sampling
        context_version = example.copy()
        context_version['force_base_retrieval'] = False  # Let it sample from [1,2,3]
        doubled_data.append(context_version)
    
    # Shuffle everything together
    random.shuffle(doubled_data)
    
    # Write output
    with open(output_file, 'w') as f:
        for example in doubled_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ Doubled from {len(data)} to {len(doubled_data)} examples")

double_training_data('train_data.jsonl', 'train_data_doubled.jsonl')