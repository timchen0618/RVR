#!/usr/bin/env python3
"""
Convert WebQSP format to QAMPARI-compatible format for evaluation.

This script converts WebQSP question-answer data into the same format as QAMPARI,
allowing it to be used with the existing evaluation code.
"""

import json
import argparse
from typing import List, Dict, Any


def convert_webqsp_to_qampari(webqsp_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert WebQSP format to QAMPARI format.
    
    Args:
        webqsp_data: List of WebQSP question dictionaries
        
    Returns:
        List of dictionaries in QAMPARI format
    """
    qampari_format = []
    
    for item in webqsp_data:
        # Extract all unique answers from all parses
        all_answers = []
        answer_entities = set()  # Track unique answers by EntityName
        
        for parse in item.get('Parses', []):
            for answer in parse.get('Answers', []):
                # Handle both Entity answers (with EntityName) and Value answers (with just AnswerArgument)
                entity_name = answer.get('EntityName')
                answer_arg = answer.get('AnswerArgument', '')
                
                # Use EntityName if available, otherwise use AnswerArgument
                answer_text = entity_name if entity_name else answer_arg
                
                # Only add if we have some answer text and haven't seen it before
                if answer_text and answer_text not in answer_entities:
                    answer_entities.add(answer_text)
                    all_answers.append({
                        'answer_text': answer_text,
                        'aid': answer_arg,
                        'aliases': [answer_text],
                        'answer_url': '',  # Not available in WebQSP
                        'proof': []  # Not available in WebQSP
                    })
        
        # Create QAMPARI-formatted entry
        qampari_entry = {
            'question_text': item.get('ProcessedQuestion', item.get('RawQuestion', '')),
            'answer_list': all_answers,
            'qid': item.get('QuestionId', ''),
            'ground_truths': []  # WebQSP doesn't have document-level ground truths
        }
        
        qampari_format.append(qampari_entry)
    
    return qampari_format


def write_jsonl(data: List[Dict], output_path: str) -> None:
    """Write list of dictionaries to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Convert WebQSP format to QAMPARI-compatible format'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file in WebQSP format'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file in QAMPARI format'
    )
    
    args = parser.parse_args()
    
    # Load WebQSP data
    print(f"Loading WebQSP data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        webqsp_data = json.load(f)
    
    # Handle both direct list and {"Questions": [...]} format
    if isinstance(webqsp_data, dict) and 'Questions' in webqsp_data:
        questions = webqsp_data['Questions']
    else:
        questions = webqsp_data
    
    print(f"Found {len(questions)} questions")
    
    # Convert to QAMPARI format
    print("Converting to QAMPARI format...")
    qampari_data = convert_webqsp_to_qampari(questions)
    
    # Write output
    print(f"Writing output to {args.output}...")
    write_jsonl(qampari_data, args.output)
    
    print(f"✓ Successfully converted {len(qampari_data)} questions")
    print(f"\nExample converted entry:")
    print(json.dumps(qampari_data[0], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()