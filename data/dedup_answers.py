#!/usr/bin/env python
"""
Deduplicate retrieved documents by answer.
For each query, ensure no two documents contain the same answer.
If multiple docs have the same answer, randomly select one.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm
import regex
import unicodedata


# ============================================================================
# Answer Matching Code (from eval script)
# ============================================================================

def _normalize(text):
    return unicodedata.normalize('NFD', text)


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def has_answer(answers: List[str], text: str, tokenizer: SimpleTokenizer) -> bool:
    """Check if a document contains an answer string using token matching."""
    text = _normalize(text)
    text_tokens = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        if not answer:
            continue
        answer = _normalize(answer)
        answer_tokens = tokenizer.tokenize(answer, uncased=True)
        if not answer_tokens:
            continue
        for i in range(0, len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[i: i + len(answer_tokens)]:
                return True
    return False


def get_all_answer_aliases(gt_item: Dict) -> List[str]:
    """
    Extract all answer aliases from ground truth item.
    Handles both 'answer_list' and 'answers' formats.
    """
    all_aliases = []
    
    if 'answer_list' in gt_item:
        for answer_obj in gt_item['answer_list']:
            if isinstance(answer_obj, dict):
                if 'aliases' in answer_obj:
                    all_aliases.extend(answer_obj['aliases'])
                if 'answer_text' in answer_obj:
                    all_aliases.append(answer_obj['answer_text'])
            elif isinstance(answer_obj, str):
                all_aliases.append(answer_obj)
    elif 'answers' in gt_item:
        if isinstance(gt_item['answers'], list):
            all_aliases = gt_item['answers']
        else:
            all_aliases = [gt_item['answers']]
    
    return [a for a in all_aliases if a and a.strip()]


def get_answer_groups(gt_item: Dict) -> List[List[str]]:
    """
    Get answer groups where each group contains aliases for one answer.
    Returns: List of answer groups, where each group is a list of aliases.
    """
    answer_groups = []
    
    if 'answer_list' in gt_item:
        for answer_obj in gt_item['answer_list']:
            group = []
            if isinstance(answer_obj, dict):
                if 'aliases' in answer_obj:
                    group.extend(answer_obj['aliases'])
                if 'answer_text' in answer_obj:
                    group.append(answer_obj['answer_text'])
            elif isinstance(answer_obj, str):
                group.append(answer_obj)
            
            if group:
                answer_groups.append(group)
    elif 'answers' in gt_item:
        if isinstance(gt_item['answers'], list):
            # Each answer is its own group
            for ans in gt_item['answers']:
                if ans and ans.strip():
                    answer_groups.append([ans])
        else:
            answer_groups.append([gt_item['answers']])
    
    return answer_groups


# ============================================================================
# Data Loading Functions
# ============================================================================

def read_jsonl(file_path):
    """Read a JSONL file, skipping malformed lines."""
    data = []
    error_count = 0
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            try:
                example = json.loads(line)
                data.append(example)
            except json.JSONDecodeError:
                error_count += 1
                print(f"Warning: Skipping malformed JSON at line {line_num} in {file_path}")
                continue
    if error_count > 0:
        print(f"Total malformed lines skipped: {error_count}")
    return data



def read_json(file_path: str) -> List[Dict]:
    """Read a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(data: List[Dict], file_path: str):
    """Write list of dictionaries to JSONL file (one compact object per line)."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_json(data: List[Dict], file_path: str):
    """Write list of dictionaries to JSON file (one compact object per line)."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ============================================================================
# Main Processing Functions
# ============================================================================

def deduplicate_by_answer(
    merged_file: str,
    ground_truth_file: str,
    output_file: str
) -> Dict:
    """
    Deduplicate retrieved documents by answer.
    Keep only one document per unique answer per query.
    """
    
    tokenizer = SimpleTokenizer()
    
    print(f"\nLoading ground truth: {ground_truth_file}")
    ground_truth = read_jsonl(ground_truth_file)
    print(f"  Loaded {len(ground_truth)} ground truth entries")
    
    print(f"\nLoading merged retriever file: {merged_file}")
    try:
        merged_data = read_json(merged_file)
        print(f"  Loaded {len(merged_data)} queries (JSON format)")
    except:
        merged_data = read_jsonl(merged_file)
        print(f"  Loaded {len(merged_data)} queries (JSONL format)")
    
    # Verify lengths match
    if len(ground_truth) != len(merged_data):
        print(f"\nWARNING: Length mismatch!")
        print(f"  Ground truth: {len(ground_truth)} examples")
        print(f"  Merged file: {len(merged_data)} examples")
        min_len = min(len(ground_truth), len(merged_data))
        print(f"  Will only process first {min_len} examples")
        ground_truth = ground_truth[:min_len]
        merged_data = merged_data[:min_len]
    
    # Statistics
    stats = {
        'n_queries': len(merged_data),
        'total_docs_before': 0,
        'total_docs_after': 0,
        'docs_removed': 0,
        'queries_with_duplicates': 0,
        'avg_docs_per_query_before': 0,
        'avg_docs_per_query_after': 0,
    }
    
    print(f"\nDeduplicating documents by answer...")
    deduplicated_data = []
    
    for idx in tqdm(range(len(merged_data))):
        gt_item = ground_truth[idx]
        merged_item = merged_data[idx]
        
        # Get answer groups (each group is aliases for one answer)
        answer_groups = get_answer_groups(gt_item)
        
        # Track which documents contain which answers
        # answer_idx -> list of (doc_idx, doc)
        answer_to_docs = {}
        for answer_idx in range(len(answer_groups)):
            answer_to_docs[answer_idx] = []
        
        # Check each document
        ctxs = merged_item.get('ctxs', [])
        stats['total_docs_before'] += len(ctxs)
        
        for doc_idx, doc in enumerate(ctxs):
            doc_text = doc.get('text', '')
            
            # Check which answer(s) this document contains
            for answer_idx, answer_aliases in enumerate(answer_groups):
                if has_answer(answer_aliases, doc_text, tokenizer):
                    answer_to_docs[answer_idx].append((doc_idx, doc))
        
        # Deduplicate: keep only one doc per answer
        kept_docs = []
        had_duplicates = False
        
        for answer_idx in range(len(answer_groups)):
            docs_for_this_answer = answer_to_docs[answer_idx]
            
            if len(docs_for_this_answer) > 1:
                had_duplicates = True
                # Randomly select one document
                selected_doc = random.choice(docs_for_this_answer)[1]
                kept_docs.append(selected_doc)
                stats['docs_removed'] += len(docs_for_this_answer) - 1
            elif len(docs_for_this_answer) == 1:
                kept_docs.append(docs_for_this_answer[0][1])
        
        if had_duplicates:
            stats['queries_with_duplicates'] += 1
        
        stats['total_docs_after'] += len(kept_docs)
        
        # Create deduplicated query object
        deduplicated_item = {
            'question': merged_item.get('question', merged_item.get('question_text', '')),
            'input': merged_item.get('input', merged_item.get('question', '')),
            'id': merged_item.get('id', idx),
            'answers': merged_item.get('answers', []),
            'ctxs': kept_docs,
            'input_negative_ctxs': merged_item.get('input_negative_ctxs', [])
        }
        
        # Preserve any other fields from original
        for key in merged_item:
            if key not in deduplicated_item:
                deduplicated_item[key] = merged_item[key]
        
        deduplicated_data.append(deduplicated_item)
    
    # Calculate averages
    stats['avg_docs_per_query_before'] = (
        stats['total_docs_before'] / stats['n_queries']
        if stats['n_queries'] > 0 else 0
    )
    stats['avg_docs_per_query_after'] = (
        stats['total_docs_after'] / stats['n_queries']
        if stats['n_queries'] > 0 else 0
    )
    
    print(f"\nSaving deduplicated data to: {output_file}")
    write_json(deduplicated_data, output_file)
    
    return stats


def print_statistics(stats: Dict):
    """Print statistics."""
    
    print("\n" + "=" * 70)
    print("DEDUPLICATION STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal queries: {stats['n_queries']:,}")
    print(f"Queries with duplicate answers: {stats['queries_with_duplicates']:,}")
    
    print(f"\nTotal docs before dedup: {stats['total_docs_before']:,}")
    print(f"Total docs after dedup: {stats['total_docs_after']:,}")
    print(f"Docs removed: {stats['docs_removed']:,}")
    
    reduction_pct = (stats['docs_removed'] / stats['total_docs_before'] * 100
                     if stats['total_docs_before'] > 0 else 0)
    print(f"Reduction: {reduction_pct:.2f}%")
    
    print(f"\nAvg docs per query before: {stats['avg_docs_per_query_before']:.2f}")
    print(f"Avg docs per query after: {stats['avg_docs_per_query_after']:.2f}")


def main():
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    merged_file = '/scratch/dq2024/diverse_retriever/data/all_model_training_file_all_golds.json'  # Output from merge_ctxs.py
    ground_truth_file = '/scratch/dq2024/diverse_retriever/train_data.jsonl'
    output_file = '/scratch/dq2024/diverse_retriever/data/all_model_training_file_all_golds_dedup_answers.json'
    
    # ========================================================================
    # Process
    # ========================================================================
    
    stats = deduplicate_by_answer(merged_file, ground_truth_file, output_file)
    
    # Print statistics
    print_statistics(stats)
    
    # ========================================================================
    # Verification
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    print(f"\nReloading deduplicated file...")
    try:
        deduped = read_json(output_file)
    except:
        deduped = read_jsonl(output_file)
    print(f"✓ Deduplicated file has {len(deduped)} queries")
    
    # Check first example
    print(f"\nSample query 0:")
    print(f"  Question: {deduped[0].get('question', '')[:100]}...")
    print(f"  # ctxs: {len(deduped[0].get('ctxs', []))}")
    
    if deduped[0].get('ctxs'):
        print(f"\n  First doc:")
        first_doc = deduped[0]['ctxs'][0]
        print(f"    id: {first_doc.get('id', '')}")
        print(f"    title: {first_doc.get('title', '')[:50]}...")
        print(f"    score: {first_doc.get('score', 0)}")
    
    print(f"\n✓ Output saved to: {output_file}")


if __name__ == "__main__":
    main()