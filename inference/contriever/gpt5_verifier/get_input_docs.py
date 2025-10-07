import json
from typing import Dict, List, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_json(data: List[Dict[str, Any]], file_path: str):
    """Save data as a JSON array"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_gold_to_negative_context(gold_file: str, original_file: str, output_file: str, max_docs: int = 3, fill_from_original: bool = True):
    """
    Convert gold documents to input_negative_ctxs format
    
    Args:
        gold_file: Path to file with gold documents (output from your verifier)
        original_file: Path to original retrieval file (for filling remaining slots)
        output_file: Path to save the converted format
        max_docs: Maximum number of documents to include (default: 3)
        fill_from_original: Whether to fill remaining slots from original retrieval (default: True)
                           If False, only uses found gold documents
    """
    
    # Load the data
    gold_data = load_jsonl(gold_file)
    original_data = load_jsonl(original_file)
    
    # Ensure both files have the same number of entries
    assert len(gold_data) == len(original_data), f"File length mismatch: {len(gold_data)} vs {len(original_data)}"
    
    converted_entries = []
    doc_count_distribution = {0: 0, 1: 0, 2: 0, 3: 0}  # Track document counts
    
    for i, (gold_entry, original_entry) in enumerate(zip(gold_data, original_data)):
        print(f"Processing entry {i+1}/{len(gold_data)}")
        
        # Create the base structure
        converted_entry = {
            "question": gold_entry.get("question", ""),
            "input": gold_entry.get("input", gold_entry.get("question", "")),
            "id": gold_entry.get("id", i),
            "answers": gold_entry.get("answers", [""]),
            "ctxs": [],  # Always empty as requested
            "input_negative_ctxs": []
        }
        
        # Get gold documents (top 3 from gold file)
        gold_docs = gold_entry.get("ctxs", [])
        selected_docs = []
        used_doc_ids = set()
        
        # Take up to max_docs gold documents
        for doc in gold_docs[:max_docs]:
            doc_copy = {
                "id": doc["id"],
                "title": doc["title"], 
                "text": doc["text"],
                "score": float(doc["score"]) if isinstance(doc["score"], str) else doc["score"]
            }
            selected_docs.append(doc_copy)
            used_doc_ids.add(doc["id"])
            print(f"  Added gold doc: {doc['title'][:50]}...")
        
        # If we need more documents to reach max_docs, fill from original retrieval
        if len(selected_docs) < max_docs and fill_from_original:
            remaining_slots = max_docs - len(selected_docs)
            print(f"  Need {remaining_slots} more docs, filling from original retrieval...")
            
            # Sort original documents by score (highest first)
            original_docs = original_entry.get("ctxs", [])
            original_docs_sorted = sorted(
                original_docs, 
                key=lambda x: float(x.get("score", 0)) if isinstance(x.get("score"), str) else x.get("score", 0), 
                reverse=True
            )
            
            # Add highest scoring non-duplicate documents
            for doc in original_docs_sorted:
                if len(selected_docs) >= max_docs:
                    break
                    
                if doc["id"] not in used_doc_ids:
                    doc_copy = {
                        "id": doc["id"],
                        "title": doc["title"],
                        "text": doc["text"], 
                        "score": float(doc["score"]) if isinstance(doc["score"], str) else doc["score"]
                    }
                    selected_docs.append(doc_copy)
                    used_doc_ids.add(doc["id"])
                    print(f"  Added filler doc: {doc['title'][:50]}...")
        elif len(selected_docs) < max_docs and not fill_from_original:
            print(f"  Only using {len(selected_docs)} gold documents (fill_from_original=False)")
        
        converted_entry["input_negative_ctxs"] = selected_docs
        converted_entries.append(converted_entry)
        
        # Track document count distribution
        doc_count = len(selected_docs)
        if doc_count <= 3:
            doc_count_distribution[doc_count] += 1
        else:
            # Handle cases with more than 3 docs (shouldn't happen with max_docs=3, but just in case)
            if doc_count not in doc_count_distribution:
                doc_count_distribution[doc_count] = 0
            doc_count_distribution[doc_count] += 1
        
        print(f"  Final count: {doc_count} documents")
    
    # Save the converted data
    save_json(converted_entries, output_file)
    
    # Calculate and display statistics
    total_entries = len(converted_entries)
    
    print(f"\nConversion complete!")
    print(f"Processed {total_entries} entries")
    print(f"Output saved to: {output_file}")
    print(f"\nDocument count distribution:")
    for count in sorted(doc_count_distribution.keys()):
        num_entries = doc_count_distribution[count]
        percentage = (num_entries / total_entries) * 100
        print(f"  {count} documents: {num_entries} entries ({percentage:.1f}%)")
    
    return doc_count_distribution

def main():
    # Configuration - edit these file paths as needed
    GOLD_FILE = "qampari_base_gpt5_nano_all_gold_docs_100.json"
    ORIGINAL_FILE = "qampari_corpus_base_finetuned_only_random_inp0_t100.json"
    MAX_DOCS = 3
    
    # Generate two files: one with filling, one without
    
    # File 1: Always has 3 documents (fills from original if needed)
    OUTPUT_FILE_FILLED = "qampari_base_finetuned_gpt5_nano_top3.json"
    print("=== Generating file with filling from original retrieval ===")
    convert_gold_to_negative_context(
        GOLD_FILE,
        ORIGINAL_FILE, 
        OUTPUT_FILE_FILLED,
        MAX_DOCS,
        fill_from_original=True
    )
    
    print("\n" + "="*60 + "\n")
    
    # File 2: Only has gold documents (no filling)
    OUTPUT_FILE_GOLD_ONLY = "qampari_base_finetuned_gpt5_nano_top0123.json"
    print("=== Generating file with only gold documents ===")
    convert_gold_to_negative_context(
        GOLD_FILE,
        ORIGINAL_FILE, 
        OUTPUT_FILE_GOLD_ONLY,
        MAX_DOCS,
        fill_from_original=False
    )

if __name__ == "__main__":
    main()