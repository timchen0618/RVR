#!/usr/bin/env python3
"""
Quick fix for WebQSP file - adds 'answers' field with empty string
This matches the qampari format for evaluation
"""

import json
import os

# Input and output file paths
input_file = '/scratch/dq2024/torch/diverse_retriever/eval/contriever/base/webqsp_contreiver_tuned_base.json'
output_file = '/scratch/dq2024/torch/diverse_retriever/eval/contriever/base/webqsp_contreiver_tuned_base_t100.json'

if not os.path.exists(input_file):
    print(f"ERROR: Input file not found: {input_file}")
    exit(1)

print(f"Processing: {input_file}")
print(f"Output to: {output_file}")
print("-" * 60)

try:
    line_count = 0
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line_count += 1
            
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Remove answer_list field if it exists
                if 'answer_list' in data:
                    del data['answer_list']
                
                # Add answers field with empty string (qampari format)
                data['answers'] = [""]
                
                # Write modified line
                outfile.write(json.dumps(data) + '\n')
                processed_count += 1
                
                # Progress update every 100 lines
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines...", flush=True)
                    
            except json.JSONDecodeError as e:
                print(f"WARNING: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"ERROR on line {line_num}: {e}")
                continue
    
    print("-" * 60)
    print(f"✓ Successfully processed {processed_count} out of {line_count} lines")
    print(f"✓ Output saved to: {output_file}")
    
    # Verify output file
    output_size = os.path.getsize(output_file)
    print(f"✓ Output file size: {output_size:,} bytes")
    
except Exception as e:
    print(f"FATAL ERROR: {e}")
    exit(1)