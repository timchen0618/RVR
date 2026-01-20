import json
import random

def double_training_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
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