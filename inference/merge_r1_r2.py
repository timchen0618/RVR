import json
from tqdm import tqdm

def read_json_file(file_path):
    """Read JSON file - handle both JSONL and JSON array formats."""
    with open(file_path, 'r') as file:
        content = file.read().strip()
        
    # Check if it's a JSON array (starts with [) or JSONL
    if content.startswith('['):
        # It's a JSON array
        return json.loads(content)
    else:
        # It's JSONL format
        data = []
        for line in content.split('\n'):
            if line.strip():
                data.append(json.loads(line))
        return data

def write_jsonl(data, file_path):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def merge_first_and_second_round(first_round_gold_path, second_round_path, output_path, target_total_docs=100):
    """
    Merge first round gold documents with second round retrieval results.
    
    Args:
        first_round_gold_path: Path to first round gold documents (input_negative_ctxs field)
        second_round_path: Path to second round retrieval results (ctxs field)
        output_path: Path to save merged results
        target_total_docs: Target total number of documents per question (default: 100)
    """
    
    # Load data
    first_round_data = read_json_file(first_round_gold_path)
    second_round_data = read_json_file(second_round_path)
    
    # Validate data lengths
    assert len(first_round_data) == len(second_round_data), f'Length mismatch: {len(first_round_data)} vs {len(second_round_data)}'
    
    merged_results = []
    
    # Statistics tracking
    stats = {
        'total_questions': 0,
        'total_gold_from_r1': 0,
        'total_docs_from_r2': 0,
        'total_overlapped_docs': 0,
        'questions_with_gold_r1': 0,
        'avg_gold_per_question': 0,
        'distribution_final_docs': {},
        'distribution_gold_counts': {}
    }
    
    for idx, (r1_entry, r2_entry) in enumerate(tqdm(zip(first_round_data, second_round_data), desc="Merging R1 gold with R2 results")):
        
        # Validate questions match
        r1_question = r1_entry.get('question', r1_entry.get('input', ''))
        r2_question = r2_entry.get('question', r2_entry.get('input', ''))
        
        # Use the second round entry as the base structure (it has the correct format)
        merged_entry = {
            'question': r2_entry.get('question', ''),
            'input': r2_entry.get('input', ''),
            'id': r2_entry.get('id', idx),
            'answers': r2_entry.get('answers', []),
            'ctxs': [],  # Will be filled with merged documents
        }
        
        # Copy any additional fields from second round entry
        for key, value in r2_entry.items():
            if key not in ['question', 'input', 'id', 'answers', 'ctxs']:
                merged_entry[key] = value
        
        # Get gold documents from first round (from input_negative_ctxs field)
        r1_gold_docs = r1_entry.get('ctxs', [])
        
        # Get all documents from second round (from ctxs field)
        r2_docs = r2_entry.get('ctxs', [])
        
        # Create set of gold document IDs to avoid duplicates
        gold_doc_ids = set()
        for doc in r1_gold_docs:
            gold_doc_ids.add(doc.get('id', ''))
        
        # Start with ALL gold documents from first round
        final_docs = r1_gold_docs.copy()
        
        # Add non-duplicate documents from second round until we reach target_total_docs
        remaining_slots = target_total_docs - len(r1_gold_docs)
        overlapped_docs_count = 0  # Track overlaps for this question
        
        if remaining_slots > 0:
            for doc in r2_docs:
                if len(final_docs) >= target_total_docs:
                    break
                    
                # Only add if it's not a duplicate of gold documents
                doc_id = doc.get('id', '')
                if doc_id not in gold_doc_ids:
                    final_docs.append(doc)
                    remaining_slots -= 1
                else:
                    # This document from R2 is a duplicate of a gold doc from R1
                    overlapped_docs_count += 1
        
        # If we have more gold docs than target, just take the first target_total_docs
        if len(final_docs) > target_total_docs:
            final_docs = final_docs[:target_total_docs]
        
        merged_entry['ctxs'] = final_docs
        merged_results.append(merged_entry)
        
        # Update statistics
        stats['total_questions'] += 1
        num_gold_r1 = len(r1_gold_docs)
        num_gold_used = min(num_gold_r1, target_total_docs)  # Actual gold docs used (capped at 100)
        num_added_from_r2 = len(final_docs) - num_gold_used  # Actual docs added from R2
        
        stats['total_gold_from_r1'] += num_gold_used
        stats['total_docs_from_r2'] += num_added_from_r2
        stats['total_overlapped_docs'] += overlapped_docs_count
        
        if num_gold_r1 > 0:
            stats['questions_with_gold_r1'] += 1
        
        # Track distribution of final document counts
        final_count = len(final_docs)
        if final_count not in stats['distribution_final_docs']:
            stats['distribution_final_docs'][final_count] = 0
        stats['distribution_final_docs'][final_count] += 1
        
        # Track distribution of gold document counts from R1
        if num_gold_r1 not in stats['distribution_gold_counts']:
            stats['distribution_gold_counts'][num_gold_r1] = 0
        stats['distribution_gold_counts'][num_gold_r1] += 1
        
        if idx < 5:  # Debug info for first few questions
            overlap_info = f", {overlapped_docs_count} overlaps" if overlapped_docs_count > 0 else ""
            print(f"Question {idx}: {num_gold_r1} gold from R1, {num_added_from_r2} added from R2, {len(final_docs)} final docs{overlap_info}")
    
    # Save merged results
    write_jsonl(merged_results, output_path)
    
    # Calculate final statistics
    stats['avg_gold_per_question'] = stats['total_gold_from_r1'] / stats['total_questions']
    
    # Print comprehensive statistics
    print(f"\n=== MERGE STATISTICS ===")
    print(f"Total questions processed: {stats['total_questions']}")
    print(f"Questions with gold docs from R1: {stats['questions_with_gold_r1']}")
    print(f"Total gold docs used from R1: {stats['total_gold_from_r1']}")
    print(f"Total docs added from R2: {stats['total_docs_from_r2']}")
    print(f"Total overlapped docs omitted from R2: {stats['total_overlapped_docs']}")
    print(f"Average gold docs per question: {stats['avg_gold_per_question']:.2f}")
    print(f"Average overlaps per question: {stats['total_overlapped_docs']/stats['total_questions']:.2f}")
    
    print(f"\nDistribution of gold docs from R1:")
    for count in sorted(stats['distribution_gold_counts'].keys()):
        num_questions = stats['distribution_gold_counts'][count]
        percentage = num_questions / stats['total_questions'] * 100
        print(f"  {count} gold docs: {num_questions} questions ({percentage:.1f}%)")
    
    print(f"\nDistribution of final document counts:")
    for count in sorted(stats['distribution_final_docs'].keys()):
        num_questions = stats['distribution_final_docs'][count]
        percentage = num_questions / stats['total_questions'] * 100
        print(f"  {count} docs: {num_questions} questions ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {output_path}")
    
    return stats


def main():
    """
    Main function to merge first round gold documents with second round results.
    Based on your file naming pattern.
    """
    
    # File mappings based on your images
    # file_pairs = [
    #     # AR (Answer Retrieval) files
    #     ('../../base/base_top3/qampari_base_finetuned_only_rand_gold_AR_100_top3.json', 'qampari_corpus_only_random_gold_AR_r1contr_r2std_100.json'),
    #     ('../../base/base_top0123/qampari_base_finetuned_only_rand_gold_AR_100_top0123.json', 'qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json'),
    #     ('../../base/base_top3/qampari_base_finetuned_only_rand_gold_AR_200_top3.json', 'qampari_corpus_only_random_gold_AR_r1contr_r2std_200.json'),
    #     ('../../base/base_top0123/qampari_base_finetuned_only_rand_gold_AR_200_top0123.json', 'qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_200.json'),
        
    #     # ER (Evidence Retrieval) files  
    #     ('../../base/base_top3/qampari_base_finetuned_only_rand_gold_ER_100_top3.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_100.json'),
    #     ('../../base/base_top0123/qampari_base_finetuned_only_rand_gold_ER_100_top0123.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_100.json'),
    #     ('../../base/base_top3/qampari_base_finetuned_only_rand_gold_ER_200_top3.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_200.json'),
    #     ('../../base/base_top0123/qampari_base_finetuned_only_rand_gold_ER_200_top0123.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_200.json'),
    # ]
    file_pairs = [
        # AR (Answer Retrieval) files
       ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/training/query_rewrite/qampari_contriever_finetuned_base_rewritten_queries_and_keywords.json'),
        #('../base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/training/query_rewrite/qampari_contriever_standard_MAX3_queries_and_keywords.json'),
                
       # ('../base_AR_gold_docs_500.json', '../qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_500.json'),
        # ('../base_AR_gold_docs_200.json', 'qampari_corpus_only_random_gold_AR_r1contr_r2std_200.json'),
        # ('../base_AR_gold_docs_200.json', 'qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_200.json'),
        
        # ER (Evidence Retrieval) files  
        # ('../base_ER_gold_docs_100.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_100.json'),
        # ('../base_ER_gold_docs_100.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_100.json'),
        # ('../base_ER_gold_docs_200.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_200.json'),
        # ('../base_ER_gold_docs_200.json', 'qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_200.json'),
        #('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/inference/qwen3/standard/max12_model/qampari_corpus_base_r1qw_r2std0-12_AR_0-1_t200.json'), 
          ]

    # file_pairs = [
    #     # AR (Answer Retrieval) files
    #     ('../../base/vanilla_top3/qampari_base_finetuned_only_random_vanilla_top3.json', 'qampari_only_random_vanilla_r1contr_r2std_top3.json'),
    #     ('../../base/vanilla_top2/qampari_base_finetuned_only_random_vanilla_top2.json', 'qampari_only_random_vanilla_r1contr_r2std_top2.json'),
    #     ('../../base/vanilla_top1/qampari_base_finetuned_only_random_vanilla_top1.json', 'qampari_only_random_vanilla_r1contr_r2std_top1.json'),

    #     ('../../base/vanilla_top3/qampari_base_finetuned_only_random_vanilla_top3.json', 'qampari_only_random_vanilla_r1contr_r2std_0123_top3.json'),
    #     ('../../base/vanilla_top2/qampari_base_finetuned_only_random_vanilla_top2.json', 'qampari_only_random_vanilla_r1contr_r2std_0123_top2.json'),
    #     ('../../base/vanilla_top1/qampari_base_finetuned_only_random_vanilla_top1.json', 'qampari_only_random_vanilla_r1contr_r2std_0123_top1.json'),
        
    # ]
    # file_pairs = [
    #     # AR (Answer Retrieval) files
    #     ('../../base/vanilla_top3/qampari_base_finetuned_only_random_vanilla_top3.json', 'qampari_only_random_vanilla_r1contr_r2std_top3.json'),
    #     ('../../base/vanilla_top2/qampari_base_finetuned_only_random_vanilla_top2.json', 'qampari_only_random_vanilla_r1contr_r2std_top2.json'),
    #     ('../../base/vanilla_top1/qampari_base_finetuned_only_random_vanilla_top1.json', 'qampari_only_random_vanilla_r1contr_r2std_top1.json'),

    #     ('../../base/vanilla_top3/qampari_base_finetuned_only_random_vanilla_top3.json', 'qampari_only_random_vanilla_r1contr_r2std_0123_top3.json'),
    #     ('../../base/vanilla_top2/qampari_base_finetuned_only_random_vanilla_top2.json', 'qampari_only_random_vanilla_r1contr_r2std_0123_top2.json'),
    #     ('../../base/vanilla_top1/qampari_base_finetuned_only_random_vanilla_top1.json', 'qampari_only_random_vanilla_r1contr_r2std_0123_top1.json'),
        
    # ]
    
    print("=== MERGING FIRST ROUND GOLD (input_negative_ctxs) WITH SECOND ROUND RESULTS ===")
    
    for r1_file, r2_file in file_pairs:
        print(f"\n--- Processing {r1_file} + {r2_file} ---")
        
        # Generate output filename
        # Extract the key part (e.g., "AR", "AR_0123", "ER_03") from the first file
        if 'AR' in r1_file:
            mode = 'AR'
        else:
            mode = 'ER'
        
        if '_0123' in r2_file:
            suffix = '_0123'
        elif '_0-3' in r2_file:
            suffix = '_0-3'
        elif '_0-6' in r2_file:
            suffix = '_0-6'
        elif '_0-12' in r2_file:
            suffix = '_0-12'
        else:
            suffix = '3'
        
        if '_100' in r2_file:
            num_docs = '100'
        elif '_t200' in r2_file:
            num_docs = 't200'
        else:
            num_docs = 't500'  # default

        if '_top3' in r2_file:
            top = '_top3'
        elif '_top2' in r2_file:
            top = '_top2'
        elif '_top1' in r2_file:
            top = '_top1'
        
        #output_file = f'merged_ALL_GOLD_r1contr_r2gru_{mode}{suffix}_{num_docs}docs.json'
        #output_file = f'merged_vanilla_r1contr_r2std{suffix}_{top}.json'
        #output_file = f'/scratch/dq2024/diverse_retriever/eval/qwen3/standard/max12_model/MERGED_ALL_GOLD_r1qw_r2std0-12_AR_0-1_200docs.json'
        output_file = "/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/merged_qampari_qwen3_finetuned_base_negated_query_retrieval.json"
        
        try:
            stats = merge_first_and_second_round(
                first_round_gold_path=r1_file,
                second_round_path=r2_file, 
                output_path=output_file,
                target_total_docs=100
            )
            
            print(f"Successfully merged: {output_file}")
            
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        except Exception as e:
            print(f"Error processing {r1_file} + {r2_file}: {e}")
    
    print(f"\nAll merge operations completed!")
    print(f"Output files: merged_N_r1contr_r2gru_[AR/ER][_suffix]_[num]docs.json")


if __name__ == "__main__":
    main()