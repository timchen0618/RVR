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
    
    file_pairs = [
     #   ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_std_finetuned_infly_0-15_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-1_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-2_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-3_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-6_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-9_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-12_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-15_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-18_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-21_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-24_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-27_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-30_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-35_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-40_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-45_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-50_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-55_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-60_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-65_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-70_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-75_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-80_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-85_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-90_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-95_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/max3_model/qampari_corpus_base_r1inf_r2std0-3_AR_0-100_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t200_promote-qwen3_0-3_retrieved.json'),
        #('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-3_test.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t200_promote-qwen3_0-12_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t500_promote-qwen3_0-3_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t500_promote-qwen3_0-6_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t500_promote-qwen3_0-12_retrieved.json'),
        #('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_base_qwen3-30B_all_gold_docs_200_vllm.json', '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_qwen3-verif_input_test.json'),
       
      # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/base/qampari_qwen3_finetuned_base_0-3_q3verif_test.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_infly_base_qwen3-30B_all_gold_docs_200.json','/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/qampari_base_finetuned_infly_0-6_q3verif_input_test.json')
       #('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-6_test.json'),  
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_base_finetuned_infly_0-3_input_test.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_base_finetuned_infly_0-6_input_test.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_base_finetuned_infly_0-12_input_test.json'),
       
        # ('/scratch/dq2024/diverse_retriever/inference/contriever/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/contriever/standard/qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t200_promote-qwen3_0-3_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t200_promote-qwen3_0-6_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t200_promote-qwen3_0-12_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t500_promote-qwen3_0-3_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t500_promote-qwen3_0-6_retrieved.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t500_promote-qwen3_0-12_retrieved.json'),
      #  ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/all_golds/qampari_base_qwen3-30B_all_gold_docs_200_vllm.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/all_golds/qampari_contriever_base_qwen3-30B_all_gold_docs_200_round_two.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/MERGED_all_gold_qampari_contreiver_round1_round2.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_std_t200_promote-qwen3_0-3_retrieved_round3_100.json')
       # ('/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_base_qwen3-30B_all_gold_docs_500_vllm.json', '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_std_t500_promote-qwen3_0-3_retrieved.json')
        
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_std_finetuned_infly_0-12_t200.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_std_finetuned_infly_0-6_t500.json'),
        # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_500.json', '/scratch/dq2024/diverse_retriever/inference/infly/standard/qampari_std_finetuned_infly_0-12_t500.json'),
      #  ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_qwen3-verif_input_test.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_finetuned_std_t200_promote-qwen3_0-6_retrieved.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json','/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_finetuned_std_t200_promote-qwen3_0-3_retrieved.json'),
        #('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_500.json','/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_finetuned_std_t500_promote-qwen3_0-12_retrieved.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_500.json','/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_finetuned_std_t500_promote-qwen3_0-6_retrieved.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_500.json','/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_finetuned_std_t500_promote-qwen3_0-3_retrieved.json')
        #('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qampari_qwen3_standard_finetuned_only_random_0-12_10k_query+keywords_retrieved.json'),
      
       ('/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v100/contriever/gold_documents/round1_all_gold_docs.json', '/scratch/dq2024/diverse_retriever/query_rewrite/contriever/contriever_query+keywords_max3_retrieved.json'),
       ('/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v100/qwen3/gold_documents/round1_all_gold_docs.json', '/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qwen3_query+keywords_max6_retrieved.json'),
       ('/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v100/infly/gold_documents/round1_all_gold_docs.json', '/scratch/dq2024/diverse_retriever/query_rewrite/infly/infly_query+keywords_max6_retrieved.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/qwen3/base/all_gold/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/query_rewrite/qwen3/qwen3_query+keywords_max6_retrieved.json'),
       # ('/scratch/dq2024/diverse_retriever/inference/infly/base/all_golds/base_AR_gold_docs_200.json', '/scratch/dq2024/diverse_retriever/query_rewrite/infly/infly_query+keywords_max6_retrieved.json')
    ]  

    
    
    print("=== MERGING FIRST ROUND GOLD (input_negative_ctxs) WITH SECOND ROUND RESULTS ===")
    
    for r1_file, r2_file in file_pairs:
        print(f"\n--- Processing {r1_file} + {r2_file} ---")
        
        if "infly" in r2_file:
            output_file = f"/scratch/dq2024/diverse_retriever/eval/infly/base/infly_query_rewrite_baseline.json"
        elif "qwen3" in r2_file:
            output_file = f"/scratch/dq2024/diverse_retriever/eval/qwen3/base/qwen3_query_rewrite_baseline.json"
        else:
            output_file = f"/scratch/dq2024/diverse_retriever/eval/contriever/base/contriever_query_rewrite_baseline.json"

        
      #  output_file = f"/scratch/dq2024/diverse_retriever/eval/infly/merged_qampari_base_finetuned_infly{suffix}_input_test.json"
      #  output_file = "/scratch/dq2024/diverse_retriever/eval/infly/qwen3_verifier/qampari_base_finetuned_infly_0-3_q3verif_input_test.json"
        #output_file = "/scratch/dq2024/diverse_retriever/eval/contriever/base/MERGED_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json"
        
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