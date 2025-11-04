#!/usr/bin/env python3
"""
Quick fix for WebQSP file - adds 'answers' field with empty string
This matches the qampari format for evaluation
"""

import json

# Input and output file paths
input_file = '/scratch/dq2024/torch/diverse_retriever/eval/contriever/base/webqsp_contreiver_tuned_base.json'
output_file = '/scratch/dq2024/torch/diverse_retriever/eval/contriever/base/webqsp_contreiver_tuned_base.json'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        
        # Remove answer_list field
        if 'answer_list' in data:
            del data['answer_list']
        
        # Add answers field with empty string (qampari format)
        data['answers'] = [""]
        
        outfile.write(json.dumps(data) + '\n')

print(f"Fixed file saved to: {output_file}")