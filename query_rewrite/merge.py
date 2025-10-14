import json
from typing import Dict, List, Any

# Configuration - Set your file names here
# INPUT_FILE1 = "qampari_qwen3_finetuned_base_query1.json"  # First JSON file
# INPUT_FILE2 = "qampari_qwen3_finetuned_base_query2.json"  # Second JSON file
# OUTPUT_FILE = "qampari_qwen3_finetuned_base_merged_query1_query2.json"  # Output file

INPUT_FILE1 = "qampari_qwen3_keywords.json"  # First JSON file
INPUT_FILE2 = "qampari_qwen3_keywords.json"  # Second JSON file
OUTPUT_FILE = "qwen3/qampari_qwen3_rewritten_original+keywords.json"  # Output file

def merge_documents_top_scoring(file1: str, file2: str, output_file: str):
    """
    Merge documents from two JSON files, keeping top 100 unique documents by score.
    """
    
    # Load data from both files
    data1 = []
    data2 = []
    
    print(f"Loading {file1}...")
    with open(file1, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data1.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in file1: {e}")
                    continue
    
    print(f"Loading {file2}...")
    with open(file2, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data2.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in file2: {e}")
                    continue
    
    print(f"Loaded {len(data1)} entries from {file1}")
    print(f"Loaded {len(data2)} entries from {file2}")
    
    # Create a mapping for easier matching
    # Assuming both files have the same number of entries in the same order
    # If not, we'll need to match by original_query or id
    
    merged_entries = []
    
    # Process each pair of entries
    for i in range(min(len(data1), len(data2))):
        entry1 = data1[i]
        entry2 = data2[i]
        
        # Verify they're for the same query (by id or original_query)
        if entry1.get('id') != entry2.get('id'):
            print(f"Warning: IDs don't match at index {i}: {entry1.get('id')} vs {entry2.get('id')}")
            # Try to match by original_query instead
            if entry1.get('original_query') != entry2.get('original_query'):
                print(f"Error: Entries at index {i} don't match. Skipping.")
                continue
        
        # Combine documents from both entries
        docs1 = entry1.get('ctxs', [])
        docs2 = entry2.get('keywords', [])
        
        print(f"Entry {i}: Combining {len(docs1)} + {len(docs2)} documents")
        
        # Create a dictionary to track unique documents by ID
        # Key: document ID, Value: (score, document)
        unique_docs = {}
        
        # Process documents from file1
        for doc in docs1:
            doc_id = doc.get('id', '')
            score = float(doc.get('score', 0))
            
            if doc_id:
                if doc_id not in unique_docs or score > unique_docs[doc_id][0]:
                    unique_docs[doc_id] = (score, doc)
        
        # Process documents from file2
        for doc in docs2:
            doc_id = doc.get('id', '')
            score = float(doc.get('score', 0))
            
            if doc_id:
                if doc_id not in unique_docs or score > unique_docs[doc_id][0]:
                    unique_docs[doc_id] = (score, doc)
        
        # Sort by score and take top 100
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x[0], reverse=True)
        top_100_docs = [doc for score, doc in sorted_docs[:100]]
        
        print(f"  → Resulted in {len(unique_docs)} unique docs, keeping top {len(top_100_docs)}")
        
        # Create merged entry
        merged_entry = {
            "question": entry1.get('original_query', ''),  # Use original_query as question
            "question_one": entry1.get('question', ''),    # Save version 1 question
            "question_two": entry2.get('question', ''),    # Save version 2 question
            "original_query": entry1.get('original_query', ''),  # Keep original_query for reference
            "input": entry1.get('input', ''),
            "id": entry1.get('id', i),
            "answers": entry1.get('answers', []),
            "ctxs": top_100_docs  # Top 100 unique documents
        }
        
        # Add optional fields if they exist
        if 'input_negative_ctxs' in entry1:
            merged_entry['input_negative_ctxs'] = entry1['input_negative_ctxs']
        
        merged_entries.append(merged_entry)
    
    # Save merged data
    print(f"\nSaving {len(merged_entries)} merged entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Successfully merged files!")
    
    # Print statistics
    if merged_entries:
        total_docs = sum(len(e.get('ctxs', [])) for e in merged_entries)
        avg_docs = total_docs / len(merged_entries)
        print(f"\nStatistics:")
        print(f"  Total merged entries: {len(merged_entries)}")
        print(f"  Average documents per entry: {avg_docs:.1f}")
        
        # Sample output for verification
        print(f"\nSample merged entry (first entry):")
        sample = merged_entries[0]
        print(f"  Original query: {sample['original_query'][:100]}...")
        print(f"  Question (output): {sample['question'][:100]}...")
        print(f"  Question one: {sample['question_one'][:100]}...")
        print(f"  Question two: {sample['question_two'][:100]}...")
        print(f"  Number of documents: {len(sample.get('ctxs', []))}")
        if sample.get('ctxs'):
            print(f"  Top document score: {sample['ctxs'][0].get('score', 'N/A')}")

def main():
    try:
        merge_documents_top_scoring(INPUT_FILE1, INPUT_FILE2, OUTPUT_FILE)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()